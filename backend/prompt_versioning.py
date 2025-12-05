"""
Prompt Version Control System
Git-like versioning for prompts with branching, merging, and history tracking
"""

import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from difflib import unified_diff
import uuid
import logging

logger = logging.getLogger(__name__)


class PromptVersion:
    """Represents a single version of a prompt"""
    
    def __init__(
        self,
        prompt_id: str,
        content: str,
        parent_id: Optional[str] = None,
        branch: str = "main",
        metadata: Optional[Dict[str, Any]] = None,
        version_id: Optional[str] = None
    ):
        self.version_id = version_id or self._generate_version_hash(content)
        self.prompt_id = prompt_id
        self.content = content
        self.parent_id = parent_id
        self.branch = branch
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc)
        self.author = self.metadata.get("author", "anonymous")
        self.message = self.metadata.get("message", "")
        self.performance_metrics = self.metadata.get("performance_metrics", {})
    
    @staticmethod
    def _generate_version_hash(content: str) -> str:
        """Generate unique version ID based on content"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "version_id": self.version_id,
            "prompt_id": self.prompt_id,
            "content": self.content,
            "parent_id": self.parent_id,
            "branch": self.branch,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "author": self.author,
            "message": self.message,
            "performance_metrics": self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptVersion':
        """Create from dictionary"""
        version = cls(
            prompt_id=data["prompt_id"],
            content=data["content"],
            parent_id=data.get("parent_id"),
            branch=data.get("branch", "main"),
            metadata=data.get("metadata", {}),
            version_id=data.get("version_id")
        )
        if isinstance(data.get("created_at"), str):
            version.created_at = datetime.fromisoformat(data["created_at"])
        return version


class PromptVersionControl:
    """Git-like version control system for prompts"""
    
    def __init__(self, db):
        """
        Initialize version control system
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.versions_collection = db.prompt_versions
        self.branches_collection = db.prompt_branches
    
    async def create_version(
        self,
        prompt_id: str,
        content: str,
        branch: str = "main",
        message: str = "",
        author: str = "anonymous",
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> PromptVersion:
        """
        Create a new version of a prompt
        
        Args:
            prompt_id: Unique prompt identifier
            content: Prompt content
            branch: Branch name (default: "main")
            message: Commit message
            author: Author identifier
            performance_metrics: Performance data (scores, metrics)
            
        Returns:
            Created PromptVersion
        """
        # Get parent version (latest in branch)
        parent = await self.get_latest_version(prompt_id, branch)
        parent_id = parent.version_id if parent else None
        
        # Calculate diff if there's a parent
        diff = None
        if parent:
            diff = self._calculate_diff(parent.content, content)
        
        # Create version
        metadata = {
            "message": message,
            "author": author,
            "performance_metrics": performance_metrics or {},
            "diff": diff,
            "diff_size": len(diff) if diff else 0
        }
        
        version = PromptVersion(
            prompt_id=prompt_id,
            content=content,
            parent_id=parent_id,
            branch=branch,
            metadata=metadata
        )
        
        # Save to database
        await self.versions_collection.insert_one(version.to_dict())
        
        # Update branch pointer
        await self._update_branch_head(prompt_id, branch, version.version_id)
        
        logger.info(f"Created version {version.version_id} for prompt {prompt_id} on branch {branch}")
        
        return version
    
    async def get_version(self, prompt_id: str, version_id: str) -> Optional[PromptVersion]:
        """Get a specific version"""
        version_data = await self.versions_collection.find_one({
            "prompt_id": prompt_id,
            "version_id": version_id
        })
        
        if version_data:
            return PromptVersion.from_dict(version_data)
        return None
    
    async def get_latest_version(
        self,
        prompt_id: str,
        branch: str = "main"
    ) -> Optional[PromptVersion]:
        """Get the latest version in a branch"""
        version_data = await self.versions_collection.find_one(
            {"prompt_id": prompt_id, "branch": branch},
            sort=[("created_at", -1)]
        )
        
        if version_data:
            return PromptVersion.from_dict(version_data)
        return None
    
    async def get_version_history(
        self,
        prompt_id: str,
        branch: Optional[str] = None,
        limit: int = 50
    ) -> List[PromptVersion]:
        """
        Get version history for a prompt
        
        Args:
            prompt_id: Prompt identifier
            branch: Optional branch filter
            limit: Maximum number of versions to return
            
        Returns:
            List of versions, newest first
        """
        query = {"prompt_id": prompt_id}
        if branch:
            query["branch"] = branch
        
        cursor = self.versions_collection.find(query).sort("created_at", -1).limit(limit)
        versions = []
        
        async for version_data in cursor:
            versions.append(PromptVersion.from_dict(version_data))
        
        return versions
    
    async def create_branch(
        self,
        prompt_id: str,
        branch_name: str,
        from_version: Optional[str] = None,
        from_branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Create a new branch
        
        Args:
            prompt_id: Prompt identifier
            branch_name: New branch name
            from_version: Version to branch from (default: latest in from_branch)
            from_branch: Branch to branch from
            
        Returns:
            Branch information
        """
        # Check if branch already exists
        existing = await self.branches_collection.find_one({
            "prompt_id": prompt_id,
            "branch": branch_name
        })
        
        if existing:
            raise ValueError(f"Branch '{branch_name}' already exists")
        
        # Get source version
        if from_version:
            source_version = await self.get_version(prompt_id, from_version)
        else:
            source_version = await self.get_latest_version(prompt_id, from_branch)
        
        if not source_version:
            raise ValueError("Source version not found")
        
        # Create branch record
        branch_data = {
            "prompt_id": prompt_id,
            "branch": branch_name,
            "head": source_version.version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_branch": from_branch,
            "source_version": source_version.version_id
        }
        
        await self.branches_collection.insert_one(branch_data)
        
        logger.info(f"Created branch '{branch_name}' for prompt {prompt_id}")
        
        return branch_data
    
    async def merge_branches(
        self,
        prompt_id: str,
        source_branch: str,
        target_branch: str,
        strategy: str = "best_performing",
        author: str = "anonymous"
    ) -> PromptVersion:
        """
        Merge one branch into another
        
        Args:
            prompt_id: Prompt identifier
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            strategy: Merge strategy ("best_performing", "latest", "manual")
            author: Merge author
            
        Returns:
            Merged version
        """
        # Get latest versions from both branches
        source_version = await self.get_latest_version(prompt_id, source_branch)
        target_version = await self.get_latest_version(prompt_id, target_branch)
        
        if not source_version or not target_version:
            raise ValueError("Source or target branch not found")
        
        # Determine merge content based on strategy
        if strategy == "best_performing":
            source_perf = source_version.performance_metrics.get("score", 0)
            target_perf = target_version.performance_metrics.get("score", 0)
            
            if source_perf > target_perf:
                merged_content = source_version.content
                message = f"Merge {source_branch} into {target_branch} (better performance)"
            else:
                merged_content = target_version.content
                message = f"Keep {target_branch} content (better performance)"
        elif strategy == "latest":
            merged_content = source_version.content
            message = f"Merge {source_branch} into {target_branch}"
        else:
            raise ValueError(f"Unsupported merge strategy: {strategy}")
        
        # Create merged version
        merged_version = await self.create_version(
            prompt_id=prompt_id,
            content=merged_content,
            branch=target_branch,
            message=message,
            author=author,
            performance_metrics=source_version.performance_metrics
        )
        
        logger.info(f"Merged {source_branch} into {target_branch} for prompt {prompt_id}")
        
        return merged_version
    
    async def rollback(
        self,
        prompt_id: str,
        target_version: str,
        branch: str = "main",
        author: str = "anonymous"
    ) -> PromptVersion:
        """
        Rollback to a previous version
        
        Args:
            prompt_id: Prompt identifier
            target_version: Version ID to rollback to
            branch: Branch to rollback on
            author: Rollback author
            
        Returns:
            New version (rollback creates a new version)
        """
        # Get target version
        old_version = await self.get_version(prompt_id, target_version)
        if not old_version:
            raise ValueError(f"Version {target_version} not found")
        
        # Create new version with old content
        rollback_version = await self.create_version(
            prompt_id=prompt_id,
            content=old_version.content,
            branch=branch,
            message=f"Rollback to version {target_version}",
            author=author,
            performance_metrics=old_version.performance_metrics
        )
        
        logger.info(f"Rolled back prompt {prompt_id} to version {target_version}")
        
        return rollback_version
    
    async def compare_versions(
        self,
        prompt_id: str,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """
        Compare two versions
        
        Args:
            prompt_id: Prompt identifier
            version_a: First version ID
            version_b: Second version ID
            
        Returns:
            Comparison data with diff
        """
        v_a = await self.get_version(prompt_id, version_a)
        v_b = await self.get_version(prompt_id, version_b)
        
        if not v_a or not v_b:
            raise ValueError("One or both versions not found")
        
        diff = self._calculate_diff(v_a.content, v_b.content)
        
        perf_a = v_a.performance_metrics.get("score", 0)
        perf_b = v_b.performance_metrics.get("score", 0)
        perf_delta = perf_b - perf_a
        
        return {
            "version_a": {
                "version_id": v_a.version_id,
                "created_at": v_a.created_at.isoformat(),
                "author": v_a.author,
                "message": v_a.message,
                "performance": perf_a
            },
            "version_b": {
                "version_id": v_b.version_id,
                "created_at": v_b.created_at.isoformat(),
                "author": v_b.author,
                "message": v_b.message,
                "performance": perf_b
            },
            "diff": diff,
            "performance_delta": perf_delta,
            "performance_improved": perf_delta > 0
        }
    
    @staticmethod
    def _calculate_diff(old_content: str, new_content: str) -> str:
        """Calculate unified diff between two versions"""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = unified_diff(
            old_lines,
            new_lines,
            fromfile="old",
            tofile="new",
            lineterm=""
        )
        
        return "".join(diff)
    
    async def _update_branch_head(
        self,
        prompt_id: str,
        branch: str,
        version_id: str
    ):
        """Update branch HEAD pointer"""
        await self.branches_collection.update_one(
            {"prompt_id": prompt_id, "branch": branch},
            {"$set": {"head": version_id, "updated_at": datetime.now(timezone.utc).isoformat()}},
            upsert=True
        )
    
    async def list_branches(self, prompt_id: str) -> List[Dict[str, Any]]:
        """List all branches for a prompt"""
        cursor = self.branches_collection.find({"prompt_id": prompt_id})
        branches = []
        
        async for branch_data in cursor:
            branches.append({
                "branch": branch_data["branch"],
                "head": branch_data["head"],
                "created_at": branch_data.get("created_at"),
                "updated_at": branch_data.get("updated_at")
            })
        
        return branches


async def get_version_control(db) -> PromptVersionControl:
    """Get version control instance"""
    return PromptVersionControl(db)
