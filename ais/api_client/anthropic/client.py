# =======================================================================
# SECTION 1: IMPORTS
# =======================================================================

"""
API client: Anthropic.
Production wrapper with cost tracking and error handling.
"""

import anthropic 
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Optional

# =======================================================================
# SECTION 2: CONSTANTS
# =======================================================================

# NO DEFAULT MODEL
"""
Philosophy: User sovereignty - You choose your Model, not us.
First-time setup will prompt user to select model explicitly.
Choice is saved to config for future sessions.
This is intentional: Own your API, own your costs, own your decisions.
"""

# Model identifiers as of 20260122 (Jan 22nd, 2026)
""" 
Using aliases for automatic updates during development.
Will switch to specific version in production (Phase 3+).
"""

## Haiku 4.5 - Fastest, lowest cost
HAIKU_MODEL = "claude-haiku-4-5"
## Current version: claude-haiku-4-5-20251001 (Oct 01st, 2025)

## Sonnet 4.5 - Best balance for most tasks
SONNET_MODEL = "claude-sonnet-4-5"
## Current version: claude-sonnet-4-5-20250929 (Sept 29th, 2025)

## Opus 4.5 - Most capable, highest cost
OPUS_MODEL = "claude-opus-4-5"
## Current version: claude-opus-4-5-20251101 (Nov 01st, 2025)

# Pricing as of 20260122 (Jan 22nd, 2026)

## Haiku 4.5: $1 input / $5 output per 1M tokens
HAIKU_INPUT = 1.00
HAIKU_OUTPUT = 5.00
HAIKU_CONTEXT = 200000

## Sonnet 4.5: $3 input / $15 output per 1M tokens
SONNET_INPUT = 3.00
SONNET_OUTPUT = 15.00
SONNET_CONTEXT = 200000

## Opus 4.5: $5 input / $25 output per 1M tokens
OPUS_INPUT = 5.00
OPUS_OUTPUT = 25.00
OPUS_CONTEXT = 200000

# =======================================================================
# SECTION 3: HELPER FUNCTIONS
# =======================================================================

def get_api_key():
    """
    Get API key from environment or guide user through permanent setup.

    Philosophy: teach users proper environment variable management and
    does not leave them stranded if new to API.

    Returns:
        str: API key
    """
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("\n"+"="*60)
        print("ERROR: ANTHROPIC_API_KEY not found in environment")
        print("="*60)
        print("\nLet's set it up properly.")
        print("This will add it to ~/.bashrc so it persists across sessions.")
        print("\nYou'll need your API key from https://console.anthropic.com/")
        print("\n"+"="*60)

        # Prompt for key
        api_key = input("Enter your Anthropic API key (starts with 'sk-ant-'): ").strip()

        # Validate format
        if not api_key.startswith("sk-ant-"):
            print("\nERROR: Invalid key format. Must start with 'sk-ant-'")
            sys.exit(1)

        # Ask if they want permanent setup
        print("\nDo you want to save this to ~/ .bashrc? (Recommended)")
        print("This will make the key available in all future terminal sessions.")
        save = input("Save permanently? (y/n): ").strip().lower()

        if save == 'y':
            # Write to bashrc
            bashrc_path = os.path.expanduser("~/.bashrc")
            with open(bashrc_path, 'a') as f:
                f.write(f"\n# Anthropic API Key (added by LLMAIS)\n")
                f.write(f"export ANTHROPIC_API_KEY='{api_key}'\n")
            
            print("\n[OK] API key saved to ~/.bashrc")
            print("\nNote: Key is active NOW for this session.")
            print("For NEW terminal sessions, run: source ~/.bashrc")
            print("(Or just close and reopen your terminal.)")
        else:
            print("\nKey will be used for this session only.")
            print("Set it permanently later with:")
            print(f" echo \"export ANTHROPIC_API_KEY='{api_key}\"' >> ~/.bashrc")
    
        print("\n"+"="*60)

    return api_key

def estimate_cost(input_tokens, output_tokens, model):
    """
    Calculate estimated cost for a query. 

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model string (e.g., "claude-sonnet-4-5")

    Returns:
        float: Estimated cost in USD
    """

    # Determine pricing based on model
    if "haiku" in model.lower():
        input_cost = (input_tokens / 1_000_000) * HAIKU_INPUT
        output_cost = (output_tokens / 1_000_000) * HAIKU_OUTPUT
    elif "sonnet" in model.lower():
        input_cost = (input_tokens / 1_000_000) * SONNET_INPUT
        output_cost = (output_tokens / 1_000_000) * SONNET_OUTPUT
    elif "opus" in model.lower():
        input_cost = (input_tokens / 1_000_000) * OPUS_INPUT
        output_cost = (output_tokens / 1_000_000) * OPUS_OUTPUT
    else:
        # Raise Error if none chosen
        raise ValueError(
            f"\nUnknown model: {model}\n"
            f"Valid models: {HAIKU_MODEL}, {SONNET_MODEL}, {OPUS_MODEL}\n"
            f"Use client.set_model() to change your model"
        )

    return input_cost + output_cost

def format_cost(cost_usd):
    """
    Format cost for display.

    Args:
        cost_usd: Cost in USD

    Returns:
        str: Formatted cost string
    """

    if cost_usd < 0.01:
        return f"${cost_usd:.4f}" # 4 decimals for tiny costs
    else:
        return f"${cost_usd:.2f}" # 2 decimals for normal costs

# =======================================================================
# SECTION 4: CLIENT CLASS
# =======================================================================

class Client:
    """
    Main client for: Anthropic API.

    Handles model selection, API calls, and cost tracking.
    User must explicitly choose model on first run.
    Choice is saved to config/api for future sessions.
    """

    def __init__(self, api_key=None):
        """
        Initialize client: Anthropic.

        Args:
            api_key: Optional API key. If not provided, calls get_api_key()
        """

        # Get API key
        self.api_key = api_key or get_api_key()
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Try to load saved model and max tokens from config
        self.model, self.default_max_tokens = self._load_model_from_config()

        # If no saved model, prompt user to select one
        if self.model is None or self.default_max_tokens is None:
            self._prompt_model_selection()

    def _load_model_from_config(self):
        """
        Load saved model and max tokens from config/api.

        Returns:
            tuple: (model, max_tokens) if found, (None,None) otherwise
        """

        config_path = "config/api/anthropic.json"

        # Check if config file exists
        if not os.path.exists(config_path):
            return None, None
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get model and max tokens from nested structure
            model = config.get('model')
            max_tokens = config.get('llm', {}).get('default_max_tokens')
            return model, max_tokens
        
        except (json.JSONDecodeError, IOError) as e:
            print(f"\n[Warning] Could not load config: {e}")
            return None, None

    def _save_model_to_config(self):
        """
        Save current model to config/api.

        Creates config directory if it doesn't exist.
        Preserves existing config settings.
        """

        config_path = "config/api/anthropic.json"
        config = {}

        # Load existing config or create new one
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                config = {}
        
        # Update llm section
        config['model'] = self.model
        config['default_max_tokens'] = self.default_max_tokens

        # Write back to file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _get_context_limit(self):
        """
        Get context limit for the current selected model.

        Returns:
            int: Context limit in tokens
        """

        if "haiku" in self.model.lower():
            return HAIKU_CONTEXT
        elif "sonnet" in self.model.lower():
            return SONNET_CONTEXT
        elif "opus" in self.model.lower():
            return OPUS_CONTEXT
        else:
            # Fallback for unknown models
            return 10000

    def _prompt_model_selection(self):
        """
        Prompt user to select a model and max_tokens limit.

        Loops until valid choice or user quits.
        Save selection to config.
        """

        print("\n"+"="*60)
        print("FIRST-TIME SETUP: Model Selection")
        print("="*60)
        print("\nPlease choose a model before using the API")
        print("Your choice will be saved for future sessions")
        print("(You can change models anytime with client.set_model())")

        # Model selection loop
        while True:
            print("\n Please select your model:")
            print(f"1. {HAIKU_MODEL} - Fastest (${HAIKU_INPUT}/${HAIKU_OUTPUT} per 1M tokens)")
            print(f"2. {SONNET_MODEL} - Balanced (${SONNET_INPUT}/${SONNET_OUTPUT} per 1M tokens)")
            print(f"3. {OPUS_MODEL} - Most Capable (${OPUS_INPUT}/${OPUS_OUTPUT} per 1M tokens)")
            print("4. Quit (no model selected)")

            choice = input("\nChoice (1/2/3/4): ").strip()

            if choice == '1':
                self.model = HAIKU_MODEL
                break
            elif choice == '2':
                self.model = SONNET_MODEL
                break
            elif choice == '3':
                self.model = OPUS_MODEL
                break
            elif choice == '4':
                print("\n[INFO] No model selected. Exiting.")
                sys.exit(0)
            else:
                print("\n[ERROR] Invalid choice.")
                print("Please enter 1, 2, 3, or 4 to quit.")

        MAX_INPUT = self._get_context_limit()

        # Max token selection loop
        print("\n"+"="*60)
        print("Max Tokens Limit (Response Length)")
        print("="*60)
        print("\nThis controls how long responses can be.")
        print("Higher = longer responses = higher cost per query.")
        print(f"Your model ({self.model}) supports up to {MAX_INPUT:,} tokens")
        print("You can override this per query if needed.")

        while True:
            print("\nSelect your default max_tokens:")
            print("1. 1024 tokens - Short answers (~750 words)")
            print("2. 2048 tokens - Medium answers (~1500 words) [RECOMMENDED]")
            print("3. 4096 tokens - Long answers (~3000 words)")
            print("4. 8192 tokens - Very long answers (~6000 words) [WARNING-HIGH COST]")
            print(f"5. Custom (enter your own value between 1 and {MAX_INPUT:,}) [ADVANCED]")

            choice = input("\nChoice (1/2/3/4/5): ").strip()

            if choice == '1':
                self.default_max_tokens = 1024
                break
            elif choice == '2':
                self.default_max_tokens = 2048
                break
            elif choice == '3':
                self.default_max_tokens = 4096
                break
            elif choice == '4':
                self.default_max_tokens = 8192
                break
            elif choice == '5':
                try:
                    custom = int(input(f"Enter max_tokens value (1-{MAX_INPUT:,}): ").strip())
                    if 1 <= custom <= MAX_INPUT:
                        # Warning for high values
                        if custom > 10000:
                            print(f"\n[WARNING] Values over 10,000 can result in high costs.")
                            print(f"At current pricing, a single query could cost ${estimate_cost(custom, custom, self.model):.2f}")
                            confirm = input("Continue? (y/n): ").strip().lower()
                            if confirm != 'y':
                                continue
                        self.default_max_tokens = custom
                        break
                    else:
                        print(f"\n[ERROR] Value must be between 1 and {MAX_INPUT:,}.")
                except ValueError:
                    print("\n[ERROR] Please enter a valid number.")
            else:
                print("\n[ERROR] Invalid choice.")
                print("Please enter 1, 2, 3, 4, or 5.")       
       
        # Save both selections
        self._save_model_to_config()

        print(f"\n[OK] Model selected: {self.model}")
        print(f"[OK] Saved to config/api/anthropic.json")
        print("="*60)

    def set_model(self, model):
        """
        Manually change the model. 

        Args:
            model: Model string (e.g., "claude-sonnet-4-5")
        """

        # Validate model
        valid_models = [HAIKU_MODEL, SONNET_MODEL, OPUS_MODEL]
        if model not in valid_models:
            raise ValueError(
                f"\nInvalid model: {model}\n"
                f"Valid models: {', '.join(valid_models)}"
            )
        
        self.model = model
        self._save_model_to_config()
        print(f"\n[OK] Model changed to : {self.model}")
        print(f"[OK] Saved to config/api/anthropic.json")

    def query(self, prompt, max_tokens=None, system_prompt=None):
        """
        Send a query to the Anthropic API. 

        Args:
            prompt: User prompt/question 
            max_tokens: Max tokens for response
            system_prompt: Optional system prompt for context

        Returns:
            dict: {
            'text': Response text,
            'input_tokens': Number of input tokens,
            'output_tokens': Number of output tokens,
            'cost': Estimated cost in USD,
            'model': Model used
            }
        
        Raises:
            ValueError: If no model selected
            anthropic.APIError: If API call fails.
        """

        # Check if model is set
        if self.model is None:
            raise ValueError(
                "\nNo model selected. Cannot make API call.\n"
                "Run client initialization again to select a model."
            )

        # Set max_tokens
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        try:
            # Build message
            if system_prompt:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            
            # Extract token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Calculate cost
            cost = estimate_cost(input_tokens, output_tokens, self.model)

            # Extract response text
            response_text = response.content[0].text

            # Build result dict
            result = {
                'text': response_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': cost,
                'model': self.model
            }

            # Output with cost included
            print("\n"+"="*60)
            print("RESPONSE:")
            print("="*60)
            print(result['text'])
            print("\n"+"="*60)
            print(f"Model: {result['model']}")
            print(f"Cost: ${result['cost']:.4f}")
            print(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
            print("="*60)

            # Return after printing
            return result

        except anthropic.APIError as e:
            print(f"\n[ERROR] API call failed: {e}")
            raise    


            


        
