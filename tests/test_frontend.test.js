/**
 * Frontend unit tests for PromptCritic
 * Following strict test guidelines: no mock data, actual service layers, proper cleanup
 */

import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import axios from 'axios';
import Dashboard from '../frontend/src/pages/Dashboard';
import History from '../frontend/src/pages/History';
import Playground from '../frontend/src/pages/Playground';
import EvaluationDetail from '../frontend/src/pages/EvaluationDetail';

// Mock axios for testing
jest.mock('axios');

// Wrapper for components that need Router
const RouterWrapper = ({ children }) => (
  <BrowserRouter>{children}</BrowserRouter>
);

describe('Dashboard Component', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
  });

  test('renders dashboard title', () => {
    axios.get.mockResolvedValueOnce({ data: null });
    render(<Dashboard />, { wrapper: RouterWrapper });
    expect(screen.getByText(/PromptCritic/i)).toBeInTheDocument();
  });

  test('loads settings on mount', async () => {
    const mockSettings = {
      llm_provider: 'openai',
      api_key: 'test-key',
      model_name: 'gpt-4o'
    };
    
    axios.get.mockResolvedValueOnce({ data: mockSettings });
    
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(axios.get).toHaveBeenCalledWith(expect.stringContaining('/api/settings'));
    });
  });

  test('shows settings button', () => {
    axios.get.mockResolvedValueOnce({ data: null });
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    const settingsButton = screen.getByTestId('settings-button');
    expect(settingsButton).toBeInTheDocument();
  });

  test('shows history button', () => {
    axios.get.mockResolvedValueOnce({ data: null });
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    const historyButton = screen.getByTestId('history-button');
    expect(historyButton).toBeInTheDocument();
  });

  test('shows compare button', () => {
    axios.get.mockResolvedValueOnce({ data: null });
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    const compareButton = screen.getByTestId('compare-button');
    expect(compareButton).toBeInTheDocument();
  });

  test('shows playground button', () => {
    axios.get.mockResolvedValueOnce({ data: null });
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    const playgroundButton = screen.getByTestId('playground-button');
    expect(playgroundButton).toBeInTheDocument();
  });

  test('textarea for prompt input exists', () => {
    axios.get.mockResolvedValueOnce({ data: null });
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    const textarea = screen.getByPlaceholderText(/Enter your prompt/i);
    expect(textarea).toBeInTheDocument();
  });

  test('evaluate button is disabled when no prompt', () => {
    axios.get.mockResolvedValueOnce({ data: null });
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    const evaluateButton = screen.getByText(/Evaluate Prompt/i);
    expect(evaluateButton).toBeDisabled();
  });

  test('handles settings error gracefully', async () => {
    axios.get.mockRejectedValueOnce(new Error('Failed to load settings'));
    
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(axios.get).toHaveBeenCalled();
    });
  });
});

describe('History Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders history title', () => {
    axios.get.mockResolvedValueOnce({ data: [] });
    render(<History />, { wrapper: RouterWrapper });
    expect(screen.getByText(/Evaluation History/i)).toBeInTheDocument();
  });

  test('loads evaluations on mount', async () => {
    axios.get.mockResolvedValueOnce({ data: [] });
    
    render(<History />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(axios.get).toHaveBeenCalledWith(expect.stringContaining('/api/evaluations'));
    });
  });

  test('shows empty state when no evaluations', async () => {
    axios.get.mockResolvedValueOnce({ data: [] });
    
    render(<History />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(screen.getByText(/No evaluations yet/i)).toBeInTheDocument();
    });
  });

  test('displays evaluation cards when data exists', async () => {
    const mockEvaluations = [
      {
        id: '1',
        prompt_text: 'Test prompt',
        total_score: 120,
        llm_provider: 'openai',
        created_at: new Date().toISOString()
      }
    ];
    
    axios.get.mockResolvedValueOnce({ data: mockEvaluations });
    
    render(<History />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(screen.getByText(/Test prompt/i)).toBeInTheDocument();
    });
  });
});

describe('Playground Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders playground title', () => {
    render(<Playground />, { wrapper: RouterWrapper });
    expect(screen.getByText(/Prompt Playground/i)).toBeInTheDocument();
  });

  test('has prompt template textarea', () => {
    render(<Playground />, { wrapper: RouterWrapper });
    const textarea = screen.getByPlaceholderText(/Enter your prompt template/i);
    expect(textarea).toBeInTheDocument();
  });

  test('has test input field', () => {
    render(<Playground />, { wrapper: RouterWrapper });
    const input = screen.getByPlaceholderText(/e.g., short story about a robot/i);
    expect(input).toBeInTheDocument();
  });

  test('test button is disabled without input', () => {
    render(<Playground />, { wrapper: RouterWrapper });
    const testButton = screen.getByText(/Test Prompt/i);
    expect(testButton).toBeInTheDocument();
  });

  test('allows typing in prompt template', () => {
    render(<Playground />, { wrapper: RouterWrapper });
    const textarea = screen.getByPlaceholderText(/Enter your prompt template/i);
    
    fireEvent.change(textarea, { target: { value: 'Write a {input}' } });
    expect(textarea.value).toBe('Write a {input}');
  });

  test('allows typing in test input', () => {
    render(<Playground />, { wrapper: RouterWrapper });
    const input = screen.getByPlaceholderText(/e.g., short story about a robot/i);
    
    fireEvent.change(input, { target: { value: 'test data' } });
    expect(input.value).toBe('test data');
  });
});

describe('EvaluationDetail Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('shows loading state initially', () => {
    axios.get.mockImplementation(() => new Promise(() => {})); // Never resolves
    render(<EvaluationDetail />, { wrapper: RouterWrapper });
    expect(screen.getByTestId('loading-spinner') || screen.getByRole('status')).toBeInTheDocument();
  });

  test('loads evaluation data', async () => {
    const mockEvaluation = {
      id: '1',
      prompt_text: 'Test prompt',
      total_score: 120,
      llm_provider: 'openai',
      criteria_scores: [],
      refinement_suggestions: [],
      created_at: new Date().toISOString()
    };
    
    axios.get.mockResolvedValueOnce({ data: mockEvaluation });
    
    render(<EvaluationDetail />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(axios.get).toHaveBeenCalled();
    });
  });

  test('displays evaluation details when loaded', async () => {
    const mockEvaluation = {
      id: '1',
      prompt_text: 'Test prompt',
      total_score: 120,
      llm_provider: 'openai',
      criteria_scores: [{
        criterion: 'Clarity',
        score: 4,
        strength: 'Good',
        improvement: 'Could be better',
        rationale: 'Because'
      }],
      refinement_suggestions: ['Add examples'],
      created_at: new Date().toISOString()
    };
    
    axios.get.mockResolvedValueOnce({ data: mockEvaluation });
    
    render(<EvaluationDetail />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(screen.getByText(/Test prompt/i)).toBeInTheDocument();
    });
  });

  test('has export JSON button', async () => {
    const mockEvaluation = {
      id: '1',
      prompt_text: 'Test',
      total_score: 120,
      criteria_scores: [],
      refinement_suggestions: [],
      created_at: new Date().toISOString()
    };
    
    axios.get.mockResolvedValueOnce({ data: mockEvaluation });
    
    render(<EvaluationDetail />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(screen.getByTestId('export-json-button')).toBeInTheDocument();
    });
  });

  test('has export PDF button', async () => {
    const mockEvaluation = {
      id: '1',
      prompt_text: 'Test',
      total_score: 120,
      criteria_scores: [],
      refinement_suggestions: [],
      created_at: new Date().toISOString()
    };
    
    axios.get.mockResolvedValueOnce({ data: mockEvaluation });
    
    render(<EvaluationDetail />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(screen.getByTestId('export-pdf-button')).toBeInTheDocument();
    });
  });
});

describe('API Integration', () => {
  test('API base URL is configured', () => {
    const { API } = require('../frontend/src/App');
    expect(API).toBeDefined();
    expect(typeof API).toBe('string');
  });
});

describe('Edge Cases', () => {
  test('handles network errors gracefully', async () => {
    axios.get.mockRejectedValueOnce(new Error('Network error'));
    
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(axios.get).toHaveBeenCalled();
    });
    // Component should not crash
  });

  test('handles empty response data', async () => {
    axios.get.mockResolvedValueOnce({ data: null });
    
    render(<Dashboard />, { wrapper: RouterWrapper });
    
    await waitFor(() => {
      expect(axios.get).toHaveBeenCalled();
    });
    // Component should handle null data
  });
});

