"""
Custom lightweight agent framework - 100% free alternative to CrewAI
"""
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import inspect

log = logging.getLogger(__name__)

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

class Agent:
    def __init__(self, role: str, goal: str, backstory: str, tools: List[Tool],
                 verbose: bool = False, allow_delegation: bool = False):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = {tool.name: tool for tool in tools}
        self.verbose = verbose
        self.allow_delegation = allow_delegation

    def execute_task(self, task: Task) -> str:
        if self.verbose:
            print(f"\n# Agent: {self.role}")
            print(f"## Task: {task.description}")

        try:
            # Robust context handling!
            context_data = {}
            if task.context:
                for ctx_task in task.context:
                    if ctx_task.result:
                        try:
                            ctx_data = json.loads(ctx_task.result)
                            if isinstance(ctx_data, dict):
                                context_data.update(ctx_data)
                            elif isinstance(ctx_data, list):
                                context_data['previous_result'] = ctx_data
                        except Exception as ex:
                            context_data['previous_result'] = ctx_task.result
            if self.verbose:
                print(f"Context keys at this step: {list(context_data.keys())}")
            result = self._intelligent_execution(task, context_data)
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

    def _intelligent_execution(self, task: Task, context: Dict) -> str:
        if "crawler" in self.role.lower() or "scan" in task.description.lower():
            if 'scan_images' in self.tools:
                import re
                dir_match = re.search(r'directory ([^\s]+)', task.description)
                directory = dir_match.group(1) if dir_match else None
                if directory:
                    return self.tools['scan_images'](directory)
                else:
                    return self.tools['scan_images']()
        elif "processor" in self.role.lower() or "embed" in task.description.lower():
            if 'embed_batch' in self.tools:
                if 'image_paths' in context:
                    return self.tools['embed_batch'](json.dumps(context))
                elif 'previous_result' in context:
                    arg = context['previous_result']
                    return self.tools['embed_batch'](
                        json.dumps({'image_paths': arg}) if isinstance(arg, list) else arg
                    )
        elif "indexer" in self.role.lower() or "index" in task.description.lower():
            if 'build_index' in self.tools:
                if 'embeddings' in context and 'metadata' in context:
                    return self.tools['build_index'](json.dumps({
                        "embeddings": context["embeddings"],
                        "metadata": context["metadata"],
                    }))
                elif 'previous_result' in context:
                    arg = context['previous_result']
                    if isinstance(arg, dict) and 'embeddings' in arg and 'metadata' in arg:
                        return self.tools['build_index'](json.dumps(arg))
                    return json.dumps({"error": "Indexing context missing embeddings/metadata."})
        elif "parser" in self.role.lower() or "parse" in task.description.lower():
            if 'parse_query' in self.tools:
                import re
                query_match = re.search(r"'([^']+)'", task.description)
                if query_match:
                    query = query_match.group(1)
                    result = self.tools['parse_query'](query)
                    print('DEBUG: parse_query result =', result)
                    return result
        elif "matcher" in self.role.lower() or "similar" in task.description.lower():
            if 'similarity_search' in self.tools:
                print(f"DEBUG matcher context: {context}")
                if 'embedding' in context:
                    return self.tools['similarity_search'](json.dumps(context))
                elif 'previous_result' in context:
                    arg = context['previous_result']
                    if isinstance(arg, dict) and 'embedding' in arg:
                        return self.tools['similarity_search'](json.dumps(arg))
                    print("DEBUG: No embedding in matcher context!")
                    return json.dumps({"error": "Search context missing embedding."})

        # -- Safe fallback: Never call with missing required args --
        if self.tools:
            tool_name = list(self.tools.keys())[0]
            sig = inspect.signature(self.tools[tool_name].func)
            required_params = [p for p in sig.parameters.values() if p.default == inspect.Parameter.empty and p.name != 'self']
            if not required_params:
                return self.tools[tool_name]()
            else:
                return json.dumps({"error": f"Tool '{tool_name}' requires arguments, but none were provided in context"})
        return json.dumps({"error": "No suitable tool found for task"})

class Crew:
    def __init__(self, agents: List[Agent], tasks: List[Task], verbose: bool = True):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose

    def kickoff(self) -> str:
        if self.verbose:
            print(f"\n🚀 Starting crew with {len(self.agents)} agents and {len(self.tasks)} tasks")
        results = []
        for i, task in enumerate(self.tasks):
            if self.verbose:
                print(f"\n📋 Executing task {i+1}/{len(self.tasks)}")
            agent = task.agent or self.agents[i % len(self.agents)]
            if agent:
                result = agent.execute_task(task)
                results.append(result)
            else:
                results.append(json.dumps({"error": f"No agent available for task: {task.description}"}))
        return results[-1] if results else json.dumps({"error": "No tasks executed"})
