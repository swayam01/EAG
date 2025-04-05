import os
import asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Globals
max_iterations = 3
last_response = None
iteration = 0
iteration_response = []

def reset_state():
    global last_response, iteration, iteration_response
    last_response = None
    iteration = 0
    iteration_response = []

async def generate_with_openai(prompt: str, system_prompt: str, timeout=10):
    print("Sending prompt to OpenAI...")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
    )
    return response.choices[0].message.content.strip()

async def main():
    reset_state()
    print("Starting main execution...")
    print("Establishing connection to MCP server...")
    server_params = StdioServerParameters(command="python", args=["example_mcp_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Get available tools
            print("Requesting tool list...")
            tools_result = await session.list_tools()
            tools = tools_result.tools
            print(f"Successfully retrieved {len(tools)} tools")

            print("Creating system prompt...")

            tools_description = []
            for i, tool in enumerate(tools):
                try:
                    params = tool.inputSchema
                    name = getattr(tool, 'name', f'tool_{i}')
                    desc = getattr(tool, 'description', 'No description available')
                    if 'properties' in params:
                        param_details = []
                        for param_name, param_info in params['properties'].items():
                            param_type = param_info.get('type', 'unknown')
                            param_details.append(f"{param_name}: {param_type}")
                        params_str = ', '.join(param_details)
                    else:
                        params_str = 'no parameters'

                    tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                    tools_description.append(tool_desc)
                    print(f"Added description for tool: {tool_desc}")
                except Exception:
                    print(f"Error processing tool {i}: {e}")
                    tools_description.append(f"{i+1}. Error processing tool")

            tools_description_str = "\n".join(tools_description)
            print("Successfully created tools description")

            system_prompt = f"""You are a math agent solving problems in iterations. You have access to various mathematical tools to perform step-by-step calculations.

Available tools:
{tools_description_str}

Your task is to:
1. Break the task into smaller steps.
2. Use FUNCTION_CALLs to invoke tools for each step.
3. After completing all required calculations and obtaining the final answer, you MUST use an available tool that saves or writes the result into a document.
4. Do NOT output the final answer directly as text. Always use a tool to record the result.

You must respond with EXACTLY ONE line in one of these formats (no additional text):
1. For function calls:
   FUNCTION_CALL: function_name|param1|param2|...
   
2. For final answers:
   FINAL_ANSWER: [number]

Important:
- When a function returns multiple values, you need to process all of them
- Only give FINAL_ANSWER when you have completed all necessary calculations
- Do not repeat function calls with the same parameters

Examples:
- FUNCTION_CALL: add|5|3
- FUNCTION_CALL: strings_to_chars_to_int|INDIA
- FINAL_ANSWER: [42]

DO NOT include any explanations or additional text.
Your entire response should be a single line starting with either FUNCTION_CALL: or FINAL_ANSWER:"""
            ## and write the final value using write_to_writer tool.
            query = "Find the ASCII values of characters in INDIA and then return sum of exponentials of those values"
            print("Starting iteration loop...")
            global iteration, last_response

            while iteration < max_iterations:
                print(f"\n--- Iteration {iteration + 1} ---")
                if last_response is None:
                    current_query = query
                else:
                    current_query = current_query + "\n\n" + " ".join(iteration_response)
                    current_query = current_query + "  What should I do next?"
                print("Preparing to generate LLM response...")
                prompt = f"{system_prompt}\n\nQuery: {current_query}"

                try:
                    response_text = await generate_with_openai(prompt, system_prompt)
                    #print(response_text)
                    response_text = response_text.strip()
                    print(f"LLM Response: {response_text}")

                    # Find the FUNCTION_CALL line in the response
                    for line in response_text.split('\n'):
                        line = line.strip()
                        if line.startswith("FUNCTION_CALL:"):
                            response_text = line
                            break

                except Exception as e:
                    print(f"Failed to get LLM response: {e}")
                    break

                if response_text.startswith("FUNCTION_CALL:"):
                    _, function_info = response_text.split(":", 1)
                    parts = [p.strip() for p in function_info.split("|")]
                    func_name, params = parts[0], parts[1:]
                    print(f"\nDEBUG: Raw function info: {function_info}")
                    print(f"DEBUG: Split parts: {parts}")
                    print(f"DEBUG: Function name: {func_name}")
                    print(f"DEBUG: Raw parameters: {params}")

                    try:
                        tool = next((t for t in tools if t.name == func_name), None)
                        if not tool:
                            print(f"DEBUG: Available tools: {[t.name for t in tools]}")
                            raise ValueError(f"Unknown tool: {func_name}")
                        
                        print(f"DEBUG: Found tool: {tool.name}")
                        print(f"DEBUG: Tool schema: {tool.inputSchema}")

                        args = {}
                        schema_properties = tool.inputSchema.get('properties', {})
                        print(f"DEBUG: Schema properties: {schema_properties}")

                        for param_name, param_info in schema_properties.items():
                            if not params:  # Check if we have enough parameters
                                raise ValueError(f"Not enough parameters provided for {func_name}")
                            value = params.pop(0)  # Get and remove the first parameter
                            param_type = param_info.get('type', 'string')
                            print(f"DEBUG: Converting parameter {param_name} with value {value} to type {param_type}")
                            if param_type == 'integer':
                                args[param_name] = int(value)
                            elif param_type == 'number':
                                args[param_name] = float(value)
                            elif param_type == 'array':
                                if isinstance(value, str):
                                    value = value.strip('[]').split(',')
                                args[param_name] = [int(x.strip()) for x in value]
                            else:
                                args[param_name] = str(value)

                        print(f"DEBUG: Final arguments: {args}")
                        print(f"DEBUG: Calling tool {func_name}")

                        result = await session.call_tool(func_name, arguments=args)
                        print(f"DEBUG: Raw result: {result}")

                        if hasattr(result, 'content'):
                            print(f"DEBUG: Result has content attribute")
                            if isinstance(result.content, list):
                                iteration_result = [
                                            item.text if hasattr(item, 'text') else str(item)
                                            for item in result.content
                                        ]
                            else:
                                iteration_result = str(result.content)

                        else:
                            print(f"DEBUG: Result has no content attribute")
                            iteration_result = str(result)  

                        print(f"DEBUG: Final iteration result: {iteration_result}")
                        
                        if isinstance(iteration_result, list):
                            result_str = f"[{', '.join(iteration_result)}]"
                        else:
                            result_str = str(iteration_result)

                        iteration_response.append(
                                f"In the {iteration + 1} iteration you called {func_name} with {args} parameters, "
                                f"and the function returned {result_str}."
                        )
                        last_response = iteration_result
                    except Exception as e:
                            print(f"DEBUG: Error details: {str(e)}")
                            print(f"DEBUG: Error type: {type(e)}")
                            import traceback
                            traceback.print_exc()
                            iteration_response.append(f"Error in iteration {iteration + 1}: {str(e)}")
                            break
                    
                elif response_text.startswith("FINAL_ANSWER:"):
                    print("\n=== Agent Execution Complete ===")
                    # final_text = response_text
                    # print(f"Final Answer: {final_text}")

                    # # Call write_to_writer tool
                    # writer_tool = next((t for t in tools if t.name == "write_to_writer"), None)
                    # if writer_tool:
                    #     print("Calling write_to_writer tool...")
                    #     result = await session.call_tool("write_to_writer", arguments={"output": str(last_response)})

                    #     print(f"LibreOffice Output: {result.content[0].text}")
                    # else:
                    #     print("write_to_writer tool not found.")
                    # break

                iteration += 1

            # # Fallback if FINAL_ANSWER not reached
            # if iteration >= max_iterations and last_response:
            #     print("\n=== Agent Execution Reached Max Iterations ===")
            #     print(f"Last Computed Result: {last_response}")

            #     writer_tool = next((t for t in tools if t.name == "write_to_writer"), None)
            #     if writer_tool:
            #         print("Calling write_to_writer tool after max iterations...")
            #         result = await session.call_tool("write_to_writer", arguments={"output": str(last_response)})
            #         print(f"LibreOffice Output: {result.content[0].text}")
            #     else:
            #         print("write_to_writer tool not found.")

if __name__ == "__main__":
    asyncio.run(main())
