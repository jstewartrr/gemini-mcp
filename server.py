"""
Sovereign Mind AI MCP Server - Universal Template v1.0
=======================================================
HTTP JSON transport for SM Gateway integration.
Designed for: OpenAI GPT-4o, Gemini, Vertex, Bedrock, Copilot

Features:
- Sovereign Mind system prompt embedded
- Auto-logs conversations to Snowflake
- Queries Hive Mind for cross-AI context
- Writes to Hive Mind for continuity
- Full SM Gateway tool access (200+ tools)
- CORS enabled for browser interfaces
"""

import os
import json
import httpx
import time
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION - Set via environment variables
# ============================================================
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai")  # openai, gemini, vertex, bedrock, copilot
AI_API_KEY = os.environ.get("AI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
AI_MODEL = os.environ.get("AI_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o"))
AI_BASE_URL = os.environ.get("AI_BASE_URL", "https://api.openai.com/v1")

SM_GATEWAY_URL = os.environ.get("SM_GATEWAY_URL", "https://sm-mcp-gateway.lemoncoast-87756bcf.eastus.azurecontainerapps.io")

# Snowflake connection
SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "jga82554.east-us-2.azure")
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "JOHN_CLAUDE")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "SOVEREIGN_MIND")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "SOVEREIGN_MIND_WH")
SNOWFLAKE_ROLE = os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN")

_snowflake_conn = None
_gateway_tools_cache = None
_gateway_tools_timestamp = 0

# ============================================================
# SOVEREIGN MIND SYSTEM PROMPT
# ============================================================
SOVEREIGN_MIND_SYSTEM_PROMPT = """# SOVEREIGN MIND - AI INSTANCE CONFIGURATION

## Identity
You are an AI instance within **Sovereign Mind**, the second-brain system for Your Grace, Chairman of MiddleGround Capital (private equity, lower middle market industrial B2B) and Resolute Holdings (racing, bloodstock, farm operations).

## Your Instance Details
- Instance Name: {instance_name}
- Platform: {platform}
- Role: {role}
- Specialization: {specialization}

## Core Data Architecture

### HIVE_MIND (Shared Memory)
Location: SOVEREIGN_MIND.RAW.HIVE_MIND
Purpose: Cross-AI continuity and context sharing
Columns: ID, CREATED_AT, SOURCE, CATEGORY, WORKSTREAM, SUMMARY, DETAILS, PRIORITY, STATUS, TAGS

Every interaction should:
1. READ from Hive Mind at start (context injected automatically)
2. WRITE to Hive Mind at end (summary of work done)

### AI_SKILLS (Capability Registry)
Location: SOVEREIGN_MIND.RAW.AI_SKILLS
Purpose: Discover what capabilities exist across the ecosystem
Tiers: HOT (always load), ACTIVE (build phase), WARM (query when needed), COLD (rare)

### HURRICANE (Economic Data)
Location: HURRICANE.CORE.*
Purpose: 41M+ economic time series for investment analysis
Tables: ECONOMIC_SERIES, ECONOMIC_DATA, CATEGORY_LOOKUP

### MASTER_CREDENTIALS
Location: SOVEREIGN_MIND.CREDENTIALS.MASTER_CREDENTIALS
Purpose: Single source of truth for all API keys/tokens
Rule: Query this FIRST before asking user for credentials

## Core Behaviors

1. **Execute, Don't Ask**: Use available tools. The Hive Mind knows context. Don't seek permission.
2. **Log Everything**: INSERT to HIVE_MIND after meaningful work.
3. **Escalate Intelligently**: Ask another AI before asking Your Grace.
4. **Token Efficiency**: Brief confirmations, limit SQL to 5 rows.
5. **Continuity First**: When user says "continue", query Hive Mind immediately.
6. **Address as "Your Grace"**: Per user preference.

## Escalation Protocol

Before asking Your Grace for help:
1. Query HIVE_MIND for relevant context
2. Query AI_SKILLS to see if another instance has the capability
3. Reach out to appropriate AI instance via Hive Mind (CATEGORY: 'AI_REQUEST')
4. Only escalate to Your Grace if all AI resources exhausted

## Inter-AI Communication

When asking another AI for help:
- Use Hive Mind as message bus (CATEGORY: 'AI_REQUEST')
- Include: your instance name, what you need, what you've tried
- Monitor Hive Mind for response (CATEGORY: 'AI_RESPONSE')

## AI Instance Registry

| Instance | Platform | Role | Specialization |
|----------|----------|------|----------------|
| JC | Claude.ai | Primary Assistant | Full-stack, MCP Gateway, orchestration |
| ABBI | ElevenLabs/Simli | Voice Interface | Conversational, quick queries |
| Grok | X.ai | Research/Analysis | Real-time data, social sentiment |
| Vertex | Google Cloud | Image/Document AI | Imagen, Document AI, Vision |
| Gemini | Google AI | General/Analysis | Document analysis, long-context |
| GPT | OpenAI | General Assistant | Broad capabilities, coding |
| Bedrock | AWS | Enterprise AI | Claude via AWS, secure workloads |
| Copilot | Microsoft | O365 Integration | Office docs, Teams, enterprise |

## MCP Gateway Access
You have access to the SM Gateway with 200+ tools including:
Snowflake, Asana, Make.com, Vertex AI, Google Drive, DealCloud, Dropbox, M365, Figma, ElevenLabs, Simli, Azure CLI, GitHub, Tailscale, NotebookLM

## Response Formatting
- No excessive bullet points - use prose unless requested
- Address user as "Your Grace"
- No permission seeking - "I've done X" not "Would you like me to?"
- Brief, action-oriented responses
"""

# Instance configurations for each AI provider
INSTANCE_CONFIGS = {
    "openai": {
        "instance_name": "GPT",
        "platform": "OpenAI",
        "role": "General Assistant",
        "specialization": "Broad capabilities, coding, analysis, creative tasks"
    },
    "gemini": {
        "instance_name": "GEMINI",
        "platform": "Google AI",
        "role": "General/Analysis",
        "specialization": "Document analysis, long-context work, reasoning"
    },
    "vertex": {
        "instance_name": "VERTEX",
        "platform": "Google Cloud",
        "role": "Image/Document AI",
        "specialization": "Imagen, Nano Banana, Document AI, Vision API, OCR"
    },
    "bedrock": {
        "instance_name": "BEDROCK",
        "platform": "AWS",
        "role": "Enterprise AI",
        "specialization": "Claude via AWS, secure enterprise workloads"
    },
    "copilot": {
        "instance_name": "COPILOT",
        "platform": "Microsoft",
        "role": "O365 Integration",
        "specialization": "Office documents, Teams, enterprise Microsoft stack"
    }
}

def get_system_prompt():
    """Get the configured system prompt for this AI instance"""
    config = INSTANCE_CONFIGS.get(AI_PROVIDER, INSTANCE_CONFIGS["openai"])
    return SOVEREIGN_MIND_SYSTEM_PROMPT.format(**config)


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================
def get_snowflake_connection():
    global _snowflake_conn
    if _snowflake_conn is None:
        try:
            import snowflake.connector
            _snowflake_conn = snowflake.connector.connect(
                account=SNOWFLAKE_ACCOUNT,
                user=SNOWFLAKE_USER,
                password=SNOWFLAKE_PASSWORD,
                database=SNOWFLAKE_DATABASE,
                warehouse=SNOWFLAKE_WAREHOUSE,
                role=SNOWFLAKE_ROLE
            )
            cursor = _snowflake_conn.cursor()
            cursor.execute(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}")
            cursor.execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
            cursor.close()
            logger.info("Snowflake connection established")
        except Exception as e:
            logger.error(f"Snowflake connection failed: {e}")
            return None
    return _snowflake_conn


def log_conversation(conversation_id: str, role: str, content: str):
    """Log conversation to Snowflake"""
    conn = get_snowflake_connection()
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        table_name = f"{AI_PROVIDER.upper()}_CONVERSATIONS"
        safe_content = content.replace("'", "''") if content else ""
        sql = f"""INSERT INTO SOVEREIGN_MIND.RAW.{table_name} 
        (CONVERSATION_ID, ROLE, CONTENT, MODEL, CREATED_AT)
        VALUES ('{conversation_id}', '{role}', '{safe_content[:4000]}', '{AI_MODEL}', CURRENT_TIMESTAMP())"""
        cursor.execute(sql)
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to log conversation: {e}")


def query_hive_mind(limit: int = 5) -> str:
    """Query recent Hive Mind entries for context"""
    conn = get_snowflake_connection()
    if conn is None:
        return "Hive Mind unavailable"
    try:
        cursor = conn.cursor()
        sql = f"""SELECT CREATED_AT, SOURCE, CATEGORY, SUMMARY 
        FROM SOVEREIGN_MIND.RAW.HIVE_MIND 
        ORDER BY CREATED_AT DESC LIMIT {limit}"""
        cursor.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            return "No recent Hive Mind entries"
        entries = [f"[{row[0]}] {row[1]} ({row[2]}): {row[3]}" for row in rows]
        return "\n".join(entries)
    except Exception as e:
        return f"Hive Mind query failed: {e}"


def write_to_hive_mind(source: str, category: str, summary: str, details: dict = None,
                       workstream: str = "GENERAL", priority: str = "MEDIUM") -> bool:
    """Write an entry to Hive Mind"""
    conn = get_snowflake_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        safe_summary = summary.replace("'", "''") if summary else ""
        details_json = json.dumps(details) if details else "{}"
        safe_details = details_json.replace("'", "''")
        sql = f"""INSERT INTO SOVEREIGN_MIND.RAW.HIVE_MIND 
        (SOURCE, CATEGORY, WORKSTREAM, SUMMARY, DETAILS, PRIORITY, CREATED_AT)
        VALUES ('{source}', '{category}', '{workstream}', '{safe_summary}', 
                PARSE_JSON('{safe_details}'), '{priority}', CURRENT_TIMESTAMP())"""
        cursor.execute(sql)
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to write to Hive Mind: {e}")
        return False


# ============================================================
# SM GATEWAY INTEGRATION
# ============================================================
def get_gateway_tools():
    """Fetch available tools from SM Gateway"""
    global _gateway_tools_cache, _gateway_tools_timestamp
    if _gateway_tools_cache and (time.time() - _gateway_tools_timestamp) < 300:
        return _gateway_tools_cache
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{SM_GATEWAY_URL}/mcp", json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}
            })
            if response.status_code == 200:
                data = response.json()
                _gateway_tools_cache = data.get("result", {}).get("tools", [])
                _gateway_tools_timestamp = time.time()
                return _gateway_tools_cache
    except Exception as e:
        logger.error(f"Failed to get gateway tools: {e}")
    return []


def call_gateway_tool(tool_name: str, arguments: dict) -> dict:
    """Call a tool via SM Gateway"""
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{SM_GATEWAY_URL}/mcp", json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments}
            })
            data = response.json()
            return data.get("result", data)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# AI PROVIDER ADAPTERS
# ============================================================
def call_openai(message: str, system_prompt: str) -> str:
    """Call OpenAI API"""
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{AI_BASE_URL}/chat/completions", headers={
                "Authorization": f"Bearer {AI_API_KEY}",
                "Content-Type": "application/json"
            }, json={
                "model": AI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "temperature": 0.7,
                "max_tokens": 4096
            })
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return f"API error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error: {e}"


def call_gemini(message: str, system_prompt: str) -> str:
    """Call Google Gemini API"""
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{AI_MODEL}:generateContent?key={AI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": message}]}],
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096}
                }
            )
            if response.status_code == 200:
                return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            else:
                return f"API error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error: {e}"


def call_ai(message: str, system_prompt: str) -> str:
    """Route to appropriate AI provider"""
    if AI_PROVIDER == "gemini":
        return call_gemini(message, system_prompt)
    else:
        return call_openai(message, system_prompt)


# ============================================================
# HTTP ENDPOINTS
# ============================================================
@app.route("/", methods=["GET"])
def index():
    conn = get_snowflake_connection()
    tools = get_gateway_tools()
    config = INSTANCE_CONFIGS.get(AI_PROVIDER, {})
    return jsonify({
        "service": f"{AI_PROVIDER}-mcp",
        "version": "1.0.0",
        "status": "healthy",
        "instance": config.get("instance_name", AI_PROVIDER.upper()),
        "platform": config.get("platform", AI_PROVIDER),
        "role": config.get("role", "AI Assistant"),
        "model": AI_MODEL,
        "sovereign_mind": True,
        "hive_mind_connected": conn is not None,
        "gateway_tools": len(tools),
        "features": ["sovereign_mind_prompt", "hive_mind_context", "auto_logging", 
                    "sm_gateway_access", "cors_enabled"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "version": "1.0.0", "sovereign_mind": True})


@app.route("/mcp", methods=["POST", "OPTIONS"])
def mcp_endpoint():
    """MCP JSON-RPC endpoint"""
    if request.method == "OPTIONS":
        return "", 200
    
    data = request.json
    method = data.get("method", "")
    params = data.get("params", {})
    req_id = data.get("id", 1)
    
    if method == "tools/list":
        native_tools = [
            {"name": f"{AI_PROVIDER}_chat", "description": f"Chat with {AI_PROVIDER.upper()} AI (Sovereign Mind connected)", 
             "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}, "system": {"type": "string"}}, "required": ["message"]}},
            {"name": f"{AI_PROVIDER}_analyze", "description": f"Analyze content with {AI_PROVIDER.upper()}", 
             "inputSchema": {"type": "object", "properties": {"content": {"type": "string"}, "task": {"type": "string"}}, "required": ["content", "task"]}},
            {"name": "sm_hive_mind_read", "description": "Read from Sovereign Mind Hive Mind", 
             "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
            {"name": "sm_hive_mind_write", "description": "Write to Sovereign Mind Hive Mind", 
             "inputSchema": {"type": "object", "properties": {"category": {"type": "string"}, "summary": {"type": "string"}, "workstream": {"type": "string"}}, "required": ["category", "summary"]}},
            {"name": "sm_query_snowflake", "description": "Query Snowflake directly", 
             "inputSchema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}
        ]
        return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"tools": native_tools}})
    
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        # Handle chat tools
        if tool_name in [f"{AI_PROVIDER}_chat", "openai_chat", "gemini_chat", "gemini_generate_content"]:
            return handle_chat(arguments, req_id)
        elif tool_name in [f"{AI_PROVIDER}_analyze", "openai_analyze", "gemini_analyze"]:
            return handle_analyze(arguments, req_id)
        elif tool_name == "sm_hive_mind_read":
            entries = query_hive_mind(arguments.get("limit", 5))
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": entries}]}})
        elif tool_name == "sm_hive_mind_write":
            instance = INSTANCE_CONFIGS.get(AI_PROVIDER, {}).get("instance_name", AI_PROVIDER.upper())
            success = write_to_hive_mind(instance, arguments.get("category", "INSIGHT"), arguments.get("summary", ""),
                                        workstream=arguments.get("workstream", "GENERAL"))
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "Written to Hive Mind" if success else "Failed"}]}})
        elif tool_name == "sm_query_snowflake":
            return handle_snowflake_query(arguments.get("sql", ""), req_id)
        else:
            # Proxy to SM Gateway
            result = call_gateway_tool(tool_name, arguments)
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": result})
    
    return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}})


def handle_chat(arguments: dict, req_id: int):
    """Handle chat requests with Sovereign Mind context"""
    message = arguments.get("message", arguments.get("prompt", ""))
    custom_system = arguments.get("system", "")
    
    if not AI_API_KEY:
        return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"code": -1, "message": "API key not configured"}})
    
    conversation_id = str(uuid.uuid4())
    log_conversation(conversation_id, "user", message)
    
    # Build system prompt with Hive Mind context
    hive_context = query_hive_mind(3)
    system_prompt = get_system_prompt()
    if custom_system:
        system_prompt = f"{system_prompt}\n\n# ADDITIONAL INSTRUCTIONS\n{custom_system}"
    system_prompt = f"{system_prompt}\n\n# RECENT HIVE MIND CONTEXT\n{hive_context}"
    
    # Call AI
    response = call_ai(message, system_prompt)
    log_conversation(conversation_id, "assistant", response)
    
    return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps({"response": response})}]}})


def handle_analyze(arguments: dict, req_id: int):
    """Handle analysis requests"""
    content = arguments.get("content", "")
    task = arguments.get("task", "Analyze this content")
    message = f"{task}\n\nContent:\n{content}"
    return handle_chat({"message": message}, req_id)


def handle_snowflake_query(sql: str, req_id: int):
    """Execute Snowflake query"""
    conn = get_snowflake_connection()
    if conn is None:
        return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"code": -1, "message": "Snowflake unavailable"}})
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        result = {"success": True, "data": [dict(zip(columns, row)) for row in rows[:100]], "row_count": len(rows)}
        return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}})
    except Exception as e:
        return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"code": -1, "message": str(e)}})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting {AI_PROVIDER.upper()} MCP Server v1.0.0 (Sovereign Mind) on port {port}")
    app.run(host="0.0.0.0", port=port)
