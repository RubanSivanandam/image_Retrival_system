"""
Custom agent framework powered by Ollama - 100% free and local
"""
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"  # Fast, efficient model

@dataclass
class Task:
    description: str
    expected_output: str
    context: Optional[List['Task']] = None
    agent: Optional['Agent'] = None
    result: Optional[str] = None

class Tool:
    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class OllamaClient:
    """Simple Ollama client for local LLM inference."""
    
    @staticmethod
    def is_available() -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
        """Generate response using Ollama."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: Ollama request failed with status {response.status_code}"
                
        except requests.RequestException as e:
            return f"Error: Failed to connect to Ollama: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

class Agent:
    def __init__(self, role: str, goal: str, backstory: str, tools: List[Tool],
                 verbose: bool = False, allow_delegation: bool = False, model: str = DEFAULT_MODEL):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = {tool.name: tool for tool in tools}
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.model = model
        self.ollama_available = OllamaClient.is_available()
        
        if not self.ollama_available and verbose:
            print(f"⚠️  Ollama not available, using rule-based fallback for {role}")

    def execute_task(self, task: Task) -> str:
        if self.verbose:
            print(f"\n# Agent: {self.role}")
            print(f"## Task: {task.description}")

        try:
            # Gather context from previous tasks
            context_data = {}
            context_summary = ""
            
            if task.context:
                for ctx_task in task.context:
                    if ctx_task.result:
                        try:
                            ctx_data = json.loads(ctx_task.result)
                            if isinstance(ctx_data, dict) and "error" not in ctx_data:
                                context_data.update(ctx_data)
                                # Create human-readable summary for LLM
                                if "image_paths" in ctx_data:
                                    context_summary += f"Found {len(ctx_data['image_paths'])} images. "
                                if "embeddings" in ctx_data:
                                    context_summary += f"Generated {len(ctx_data['embeddings'])} embeddings. "
                                if "results" in ctx_data:
                                    context_summary += f"Found {len(ctx_data['results'])} search results. "
                            elif isinstance(ctx_data, dict) and "error" in ctx_data:
                                print(f"⚠️  Previous task error: {ctx_data['error']}")
                                return json.dumps({"error": f"Previous task failed: {ctx_data['error']}"})
                        except:
                            context_data['previous_result'] = ctx_task.result
            
            if self.verbose and context_summary:
                print(f"Context: {context_summary}")
            
            # Execute task using Ollama or fallback
            result = self._execute_with_ai(task, context_data, context_summary)
            
            # Validate result
            try:
                result_data = json.loads(result)
                if isinstance(result_data, dict) and "error" in result_data:
                    if self.verbose:
                        print(f"❌ Task failed: {result_data['error']}")
                    return result
            except:
                pass
            
            task.result = result
            if self.verbose:
                print("✅ Task completed successfully")
            return result

        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            log.error(error_msg)
            if self.verbose:
                print(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})

    def _execute_with_ai(self, task: Task, context_data: Dict, context_summary: str) -> str:
        """Execute task using Ollama AI or intelligent fallback."""
        
        if self.ollama_available:
            return self._execute_with_ollama(task, context_data, context_summary)
        else:
            return self._execute_with_fallback(task, context_data)

    def _execute_with_ollama(self, task: Task, context_data: Dict, context_summary: str) -> str:
        """Execute task using Ollama for intelligent reasoning."""
        
        # Create tools description for LLM
        tools_desc = ""
        for name, tool in self.tools.items():
            tools_desc += f"- {name}: {tool.description}\n"
        
        # Construct prompt for LLM
        prompt = f"""You are a {self.role}.
Your goal: {self.goal}
Background: {self.backstory}

Available tools:
{tools_desc}

Current task: {task.description}
Context: {context_summary}

Based on the task and context, determine which tool to use and with what parameters.

Respond in this exact JSON format:
{{"tool_name": "tool_to_use", "parameters": {{"param1": "value1"}}, "reasoning": "why this tool"}}

If you need to use context data, the available context keys are: {list(context_data.keys())}

Remember:
- For scanning images: use scan_images tool
- For generating embeddings: use embed_batch tool with image_paths from context
- For building index: use build_index tool with embeddings and metadata from context  
- For parsing queries: use parse_query tool with the query text
- For similarity search: use similarity_search tool with embedding from context

Think step by step and choose the right tool for this specific task."""

        # Get LLM response
        llm_response = OllamaClient.generate(prompt, self.model)
        
        try:
            # Parse LLM response
            decision = json.loads(llm_response)
            tool_name = decision.get("tool_name")
            parameters = decision.get("parameters", {})
            reasoning = decision.get("reasoning", "")
            
            if self.verbose and reasoning:
                print(f"🤖 AI Reasoning: {reasoning}")
            
            # Execute the chosen tool
            if tool_name in self.tools:
                return self._execute_tool(tool_name, parameters, context_data)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
                
        except json.JSONDecodeError:
            if self.verbose:
                print(f"⚠️  LLM response not valid JSON, using fallback")
            return self._execute_with_fallback(task, context_data)

    def _execute_tool(self, tool_name: str, parameters: Dict, context_data: Dict) -> str:
        """Execute a specific tool with parameters."""
        tool = self.tools[tool_name]
        
        try:
            # Prepare arguments based on tool and parameters
            if tool_name == "scan_images":
                directory = parameters.get("directory")
                if directory:
                    return tool(directory)
                else:
                    return tool()
                    
            elif tool_name == "embed_batch":
                if "image_paths" in context_data:
                    return tool(json.dumps(context_data))
                else:
                    return json.dumps({"error": "No image paths in context"})
                    
            elif tool_name == "build_index":
                if "embeddings" in context_data and "metadata" in context_data:
                    return tool(json.dumps({
                        "embeddings": context_data["embeddings"],
                        "metadata": context_data["metadata"]
                    }))
                else:
                    return json.dumps({"error": "Missing embeddings or metadata in context"})
                    
            elif tool_name == "parse_query":
                query = parameters.get("query", "")
                if query:
                    return tool(query)
                else:
                    return json.dumps({"error": "No query provided"})
                    
            elif tool_name == "similarity_search":
                if "embedding" in context_data:
                    return tool(json.dumps(context_data))
                else:
                    return json.dumps({"error": "No embedding in context"})
            else:
                return json.dumps({"error": f"Unknown tool execution pattern for {tool_name}"})
                
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

    def _execute_with_fallback(self, task: Task, context_data: Dict) -> str:
        """Fallback rule-based execution when Ollama is unavailable."""
        
        # Rule-based tool selection (same as before)
        if "crawler" in self.role.lower() or "scan" in task.description.lower():
            if 'scan_images' in self.tools:
                return self.tools['scan_images']()
        elif "processor" in self.role.lower() or "embed" in task.description.lower():
            if 'embed_batch' in self.tools and 'image_paths' in context_data:
                return self.tools['embed_batch'](json.dumps(context_data))
        elif "indexer" in self.role.lower() or "index" in task.description.lower():
            if 'build_index' in self.tools and 'embeddings' in context_data and 'metadata' in context_data:
                return self.tools['build_index'](json.dumps({
                    "embeddings": context_data["embeddings"], 
                    "metadata": context_data["metadata"]
                }))
        elif "parser" in self.role.lower() or "parse" in task.description.lower():
            if 'parse_query' in self.tools:
                import re
                query_match = re.search(r"'([^']+)'", task.description)
                if query_match:
                    return self.tools['parse_query'](query_match.group(1))
        elif "matcher" in self.role.lower() or "similar" in task.description.lower():
            if 'similarity_search' in self.tools and 'embedding' in context_data:
                return self.tools['similarity_search'](json.dumps(context_data))
        
        return json.dumps({"error": "Could not determine appropriate tool for task"})

class Crew:
    def __init__(self, agents: List[Agent], tasks: List[Task], verbose: bool = True):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose

    def kickoff(self) -> str:
        if self.verbose:
            print(f"\n🚀 Starting AI crew with {len(self.agents)} agents and {len(self.tasks)} tasks")
            if OllamaClient.is_available():
                print("🤖 Using Ollama for AI reasoning")
            else:
                print("⚠️  Ollama unavailable, using rule-based fallback")
        
        results = []
        for i, task in enumerate(self.tasks):
            if self.verbose:
                print(f"\n📋 Executing task {i+1}/{len(self.tasks)}")
            
            agent = task.agent or self.agents[i % len(self.agents)]
            if agent:
                result = agent.execute_task(task)
                results.append(result)
                
                # Check for errors
                try:
                    result_data = json.loads(result)
                    if isinstance(result_data, dict) and "error" in result_data:
                        if self.verbose:
                            print(f"❌ Task {i+1} failed, stopping execution")
                            print(f"Error: {result_data['error']}")
                        return result
                except:
                    pass
            else:
                error_result = json.dumps({"error": f"No agent for task: {task.description}"})
                results.append(error_result)
                return error_result
                
        return results[-1] if results else json.dumps({"error": "No tasks executed"})
