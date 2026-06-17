import asyncio
import json
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any, List
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError

class OpenAIClient:
    """Async OpenAI client with cancellation support."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 90,
        api_version: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        wire_api: str = "responses",
        reasoning_effort: str = "xhigh",
        text_verbosity: str = "",
        store_responses: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}
        self.wire_api = wire_api if wire_api in {"chat", "responses"} else "responses"
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.store_responses = store_responses
        
        # Prepare default headers
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "claude-proxy/1.0.0"
        }
        
        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}
        
        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                default_headers=all_headers
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=all_headers
            )
        self.active_requests: Dict[str, asyncio.Event] = {}
    
    async def create_chat_completion(self, request: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        try:
            # Create task that can be cancelled
            completion_task = asyncio.create_task(
                self._create_completion(request)
            )
            
            if request_id:
                # Wait for either completion or cancellation
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [completion_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check if request was cancelled
                if cancel_task in done:
                    completion_task.cancel()
                    raise HTTPException(status_code=499, detail="Request cancelled by client")
                
                completion = await completion_task
            else:
                completion = await completion_task
            
            # Convert to dict format that matches the original interface
            if isinstance(completion, dict):
                return completion
            return completion.model_dump()
        
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def create_chat_completion_stream(self, request: Dict[str, Any], request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        try:
            streaming_completion = self._create_completion_stream(request)
            
            async for line in streaming_completion:
                # Check for cancellation before yielding each chunk
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(status_code=499, detail="Request cancelled by client")

                yield line
            
            # Signal end of stream
            yield "data: [DONE]"
                
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()
        
        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or "country, region, or territory not supported" in error_str:
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."
        
        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."
        
        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
        
        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
        
        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."
        
        # Default: return original message
        return str(error_detail)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False

    async def _create_completion(self, request: Dict[str, Any]) -> Any:
        if self.wire_api == "responses":
            response_request = self._chat_request_to_responses_request(request)
            response = await self.client.responses.create(**response_request)
            return self._responses_response_to_chat_completion(response.model_dump(), request)

        return await self.client.chat.completions.create(**request)

    async def _create_completion_stream(self, request: Dict[str, Any]) -> AsyncGenerator[str, None]:
        if self.wire_api == "responses":
            response_request = self._chat_request_to_responses_request(request)
            response_request["stream"] = True
            async for event in await self.client.responses.create(**response_request):
                for chunk in self._responses_stream_event_to_chat_chunks(event.model_dump(), request):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}"
            return

        request["stream"] = True
        if "stream_options" not in request:
            request["stream_options"] = {}
        request["stream_options"]["include_usage"] = True

        async for chunk in await self.client.chat.completions.create(**request):
            chunk_dict = chunk.model_dump()
            chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
            yield f"data: {chunk_json}"

    def _chat_request_to_responses_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        messages = request.get("messages", [])
        response_request: Dict[str, Any] = {
            "model": request["model"],
            "input": self._chat_messages_to_responses_input(messages),
            "max_output_tokens": request.get("max_tokens"),
            "store": self.store_responses,
        }

        instructions = self._collect_system_instructions(messages)
        if instructions:
            response_request["instructions"] = instructions

        if self.reasoning_effort:
            response_request["reasoning"] = {"effort": self.reasoning_effort}

        if self.text_verbosity:
            response_request["text"] = {"verbosity": self.text_verbosity}

        if request.get("tools"):
            response_request["tools"] = self._chat_tools_to_responses_tools(request["tools"])

        tool_choice = request.get("tool_choice")
        if tool_choice:
            response_request["tool_choice"] = self._chat_tool_choice_to_responses_tool_choice(tool_choice)

        return {k: v for k, v in response_request.items() if v is not None}

    def _collect_system_instructions(self, messages: List[Dict[str, Any]]) -> str:
        instructions = []
        for message in messages:
            if message.get("role") == "system":
                text = self._content_to_text(message.get("content"))
                if text:
                    instructions.append(text)
        return "\n\n".join(instructions)

    def _chat_messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.get("tool_call_id", ""),
                        "output": self._content_to_text(message.get("content")),
                    }
                )
                continue
            if role == "assistant" and message.get("tool_calls"):
                for tool_call in message.get("tool_calls", []):
                    function_data = tool_call.get("function", {})
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.get("id", ""),
                            "name": function_data.get("name", ""),
                            "arguments": function_data.get("arguments", "{}"),
                        }
                    )
                if message.get("content"):
                    items.append(
                        {
                            "role": "assistant",
                            "content": self._responses_input_content(message.get("content")),
                        }
                    )
                continue

            items.append(
                {
                    "role": role,
                    "content": self._responses_input_content(message.get("content")),
                }
            )
        return items

    def _responses_input_content(self, content: Any) -> Any:
        if isinstance(content, list):
            converted = []
            for item in content:
                if item.get("type") == "image_url":
                    converted.append(
                        {
                            "type": "input_image",
                            "image_url": item.get("image_url", {}).get("url", ""),
                        }
                    )
                elif item.get("type") == "text":
                    converted.append({"type": "input_text", "text": item.get("text", "")})
            return converted
        return content or ""

    def _chat_tools_to_responses_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        response_tools = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function_data = tool.get("function", {})
            response_tools.append(
                {
                    "type": "function",
                    "name": function_data.get("name", ""),
                    "description": function_data.get("description", ""),
                    "parameters": function_data.get("parameters", {}),
                }
            )
        return response_tools

    def _chat_tool_choice_to_responses_tool_choice(self, tool_choice: Any) -> Any:
        if isinstance(tool_choice, str):
            return tool_choice
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            return {
                "type": "function",
                "name": tool_choice.get("function", {}).get("name", ""),
            }
        return "auto"

    def _responses_response_to_chat_completion(
        self, response: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        content_parts = []
        tool_calls = []
        for item in response.get("output", []) or []:
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content", []) or []:
                    if part.get("type") in {"output_text", "text"} and part.get("text"):
                        content_parts.append(part["text"])
            elif item_type == "function_call":
                tool_calls.append(
                    {
                        "id": item.get("call_id") or item.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                )

        finish_reason = "tool_calls" if tool_calls else self._responses_finish_reason(response)
        message: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts) or None}
        if tool_calls:
            message["tool_calls"] = tool_calls

        usage = response.get("usage", {}) or {}
        return {
            "id": response.get("id", ""),
            "object": "chat.completion",
            "model": response.get("model", request.get("model", "")),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_tokens_details": usage.get("input_tokens_details", {}),
                "completion_tokens_details": usage.get("output_tokens_details", {}),
            },
        }

    def _responses_finish_reason(self, response: Dict[str, Any]) -> str:
        incomplete_details = response.get("incomplete_details") or {}
        if incomplete_details.get("reason") == "max_output_tokens":
            return "length"
        return "stop"

    def _responses_stream_event_to_chat_chunks(
        self, event: Dict[str, Any], request: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            return [
                {
                    "id": event.get("item_id", ""),
                    "object": "chat.completion.chunk",
                    "model": request.get("model", ""),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": event.get("delta", "")},
                            "finish_reason": None,
                        }
                    ],
                }
            ]
        if event_type == "response.output_item.done":
            item = event.get("item", {}) or {}
            if item.get("type") == "function_call":
                return [
                    {
                        "id": item.get("id", ""),
                        "object": "chat.completion.chunk",
                        "model": request.get("model", ""),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": item.get("call_id", ""),
                                            "type": "function",
                                            "function": {
                                                "name": item.get("name", ""),
                                                "arguments": item.get("arguments", "{}"),
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                ]
        if event_type == "response.completed":
            response = event.get("response", {}) or {}
            usage = response.get("usage")
            finish_reason = (
                "tool_calls"
                if self._responses_output_has_function_call(response)
                else self._responses_finish_reason(response)
            )
            return [
                {
                    "id": response.get("id", ""),
                    "object": "chat.completion.chunk",
                    "model": response.get("model", request.get("model", "")),
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                    "usage": self._responses_usage_to_chat_usage(usage) if usage else None,
                }
            ]
        return []

    def _responses_usage_to_chat_usage(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "prompt_tokens_details": usage.get("input_tokens_details", {}),
            "completion_tokens_details": usage.get("output_tokens_details", {}),
        }

    def _responses_output_has_function_call(self, response: Dict[str, Any]) -> bool:
        return any(
            item.get("type") == "function_call" for item in response.get("output", []) or []
        )

    def _content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in {"text", "input_text", "output_text"}:
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)
