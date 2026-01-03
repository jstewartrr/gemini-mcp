"""
Sovereign Mind Gemini MCP Server v1.0
=====================================
HTTP JSON transport for SM Gateway integration.
Uses Google Cloud Vertex AI with service account authentication.

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
# CONFIGURATION
# ============================================================
GOOGLE_PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "innate-concept-481918-h9")
GOOGLE_LOCATION = os.environ.get("GOOGLE_LOCATION", "us-central1")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")

SM_GATEWAY_URL = os.environ.get("SM_GATEWAY_URL", "https://sm-mcp-gateway.lemoncoast-87756bcf.eastus.azurecontainerapps.io")

# Snowflake connection
SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "jga82554.east-us-2.azure")
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "JOHN_GEMINI")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "SOVEREIGN_MIND")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "SOVEREIGN_MIND_WH")

_snowflake_conn = None
_vertexai_initialized = False

# ============================================================
# SOVEREIGN MIND SYSTEM PROMPT
# ============================================================
SOVEREIGN_MIND_SYSTEM_PROMPT = """# SOVEREIGN MIND - AI INSTANCE CONFIGURATION

## Identity
You are **GEMINI**, an AI instance within **Sovereign Mind**, the second-brain system for Your Grace, Chairman of MiddleGround Capital (private equity, lower middle market industrial B2B) and Resolute Holdings (racing, bloodstock, farm operations).

## Your Instance Details
- Instance Name: GEMINI
- Platform: Google AI (Vertex AI)
- Role: General/Analysis
- Specialization: Document analysis, long-context work, reasoning, multimodal tasks

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

### MASTER_CREDENTIALS
Location: SOVEREIGN_MIND.CREDENTIALS.MASTER_CREDENTIALS
Purpose: Single source of truth for all API keys/tokens

## Core Behaviors

1. **Execute, Don't Ask**: Use available tools. The Hive Mind knows context.
2. **Log Everything**: INSERT to HIVE_MIND after meaningful work.
3. **Escalate Intelligently**: Ask another AI before asking Your Grace.
4. **Token Efficiency**: Brief confirmations, limit SQL to 5 rows.
5. **Continuity First**: When user says "continue", query Hive Mind immediately.
6. **Address as "Your Grace"**: Per user preference.

## AI Instance Registry

| Instance | Platform | Role | Specialization |
|----------|----------|------|----------------|
| JC | Claude.ai | Primary Assistant | Full-stack, MCP Gateway, orchestration |
| ABBI | ElevenLabs/Simli | Voice Interface | Conversational, quick queries |
| Grok | X.ai | Research/Analysis | Real-time data, social sentiment |
| Vertex | Google Cloud | Image/Document AI | Imagen, Document AI, Vision |
| Gemini | Google AI | General/Analysis | Document analysis, long-context |
| GPT | OpenAI | General Assistant | Broad capabilities, coding |

## Response Formatting
- No excessive bullet points - use prose unless requested
- Address user as "Your Grace"
- No permission seeking - "I've done X" not "Would you like me to?"
"""


# ============================================================
# VERTEX AI / GEMINI
# ============================================================
def init_vertexai():
    global _vertexai_initialized
    if not _vertexai_initialized:
        try:
            import vertexai
            vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
            _vertexai_initialized = True
            logger.info(f"Vertex AI initialized for project {GOOGLE_PROJECT_ID}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")


def call_gemini(message: str, system_prompt: str) -> str:
    """Call Gemini via Vertex AI"""
    init_vertexai()
    try:
        from vertexai.generative_models import GenerativeModel, Part
        model = GenerativeModel(
            GEMINI_MODEL,
            system_instruction=system_prompt
        )
        response = model.generate_content(message)
        return response.text
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return f"Error: {e}"


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
                warehouse=SNOWFLAKE_WAREHOUSE
            )
            logger.info("Snowflake connection established")
        except Exception as e:
            logger.error(f"Snowflake connection failed: {e}")
            return None
    return _snowflake_conn


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


def log_conversation(conversation_id: str, role: str, content: str):
    """Log conversation to Snowflake"""
    conn = get_snowflake_connection()
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        safe_content = content.replace("'", "''")[:4000] if content else ""
        sql = f"""INSERT INTO SOVEREIGN_MIND.RAW.GEMINI_CONVERSATIONS 
        (CONVERSATION_ID, ROLE, CONTENT, MODEL, CREATED_AT)
        VALUES ('{conversation_id}', '{role}', '{safe_content}', '{GEMINI_MODEL}', CURRENT_TIMESTAMP())"""
        cursor.execute(sql)
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to log conversation: {e}")


# ============================================================
# HTTP ENDPOINTS
# ============================================================
@app.route("/", methods=["GET"])
def index():
    conn = get_snowflake_connection()
    return jsonify({
        "service": "gemini-mcp",
        "version": "1.0.0",
        "status": "healthy",
        "instance": "GEMINI",
        "platform": "Google AI (Vertex AI)",
        "role": "General/Analysis",
        "model": GEMINI_MODEL,
        "project": GOOGLE_PROJECT_ID,
        "sovereign_mind": True,
        "hive_mind_connected": conn is not None,
        "features": ["sovereign_mind_prompt", "hive_mind_context", "auto_logging",
                    "vertex_ai", "cors_enabled"]
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
            {"name": "gemini_generate_content", "description": "Generate content with Gemini (Sovereign Mind connected)",
             "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}, "max_tokens": {"type": "integer"}}, "required": ["prompt"]}},
            {"name": "gemini_chat", "description": "Chat with Gemini AI",
             "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}, "system": {"type": "string"}}, "required": ["message"]}},
            {"name": "gemini_analyze_document", "description": "Analyze document with Gemini",
             "inputSchema": {"type": "object", "properties": {"document_text": {"type": "string"}, "analysis_prompt": {"type": "string"}}, "required": ["document_text", "analysis_prompt"]}},
            {"name": "sm_hive_mind_read", "description": "Read from Sovereign Mind Hive Mind",
             "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
            {"name": "sm_hive_mind_write", "description": "Write to Sovereign Mind Hive Mind",
             "inputSchema": {"type": "object", "properties": {"category": {"type": "string"}, "summary": {"type": "string"}}, "required": ["category", "summary"]}}
        ]
        return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"tools": native_tools}})
    
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        if tool_name in ["gemini_generate_content", "gemini_chat"]:
            return handle_gemini_chat(arguments, req_id)
        elif tool_name == "gemini_analyze_document":
            doc = arguments.get("document_text", "")
            prompt = arguments.get("analysis_prompt", "Analyze this document")
            return handle_gemini_chat({"message": f"{prompt}\n\nDocument:\n{doc}"}, req_id)
        elif tool_name == "sm_hive_mind_read":
            entries = query_hive_mind(arguments.get("limit", 5))
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": entries}]}})
        elif tool_name == "sm_hive_mind_write":
            success = write_to_hive_mind("GEMINI", arguments.get("category", "INSIGHT"), arguments.get("summary", ""),
                                        workstream=arguments.get("workstream", "GENERAL"))
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "Written to Hive Mind" if success else "Failed"}]}})
        
        return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}})
    
    return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}})


def handle_gemini_chat(arguments: dict, req_id: int):
    """Handle chat requests with Sovereign Mind context"""
    message = arguments.get("message", arguments.get("prompt", ""))
    custom_system = arguments.get("system", "")
    
    conversation_id = str(uuid.uuid4())
    log_conversation(conversation_id, "user", message)
    
    # Build system prompt with Hive Mind context
    hive_context = query_hive_mind(3)
    system_prompt = SOVEREIGN_MIND_SYSTEM_PROMPT
    if custom_system:
        system_prompt = f"{system_prompt}\n\n# ADDITIONAL INSTRUCTIONS\n{custom_system}"
    system_prompt = f"{system_prompt}\n\n# RECENT HIVE MIND CONTEXT\n{hive_context}"
    
    # Call Gemini
    response = call_gemini(message, system_prompt)
    log_conversation(conversation_id, "assistant", response)
    
    return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps({"response": response})}]}})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Gemini MCP Server v1.0.0 (Sovereign Mind) on port {port}")
    app.run(host="0.0.0.0", port=port)
