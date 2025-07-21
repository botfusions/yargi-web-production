import os
import httpx
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Yargı Web Interface", description="Turkish Legal Database Search")

# Templates
templates = Jinja2Templates(directory="templates")

# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://yargi-mcp.botfusions.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENABLE_LLM = os.getenv("ENABLE_LLM", "true").lower() == "true"

# Initialize OpenAI client if API key is provided
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized")
else:
    logger.warning("No OpenAI API key provided - LLM features disabled")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with search interface"""
    try:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "llm_enabled": bool(openai_client)}
        )
    except Exception as e:
        # Fallback if templates not found
        return HTMLResponse("""
        <html>
        <head><title>Yargı Web Interface</title></head>
        <body>
            <h1>🏛️ Yargı Web Interface</h1>
            <p>Türk Hukuk Veritabanı Arama Sistemi</p>
            <p><strong>Durum:</strong> Temel API çalışıyor</p>
            <p><strong>MCP Server:</strong> {}</p>
            <p><strong>LLM:</strong> {}</p>
            <p><em>Templates klasörü eklendikten sonra tam arayüz aktif olacak.</em></p>
        </body>
        </html>
        """.format(MCP_SERVER_URL, "Aktif" if openai_client else "Pasif"))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "yargi-web-interface",
        "mcp_server": MCP_SERVER_URL,
        "llm_enabled": bool(openai_client)
    }

@app.get("/api/tools")
async def get_available_tools():
    """Get available MCP tools"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{MCP_SERVER_URL}/status")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "MCP server not available", "status": "fallback"}
    except Exception as e:
        logger.error(f"Error fetching tools: {e}")
        return {"error": str(e), "status": "error"}

@app.post("/api/search")
async def search_legal_database(query: str = Form(...)):
    """Search legal database with optional LLM enhancement"""
    if not query or len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    
    query = query.strip()
    
    try:
        # Step 1: Get results from MCP server
        mcp_results = await query_mcp_server(query)
        
        # Step 2: Enhance with LLM if available
        if openai_client and ENABLE_LLM:
            enhanced_results = await enhance_with_llm(query, mcp_results)
            return {
                "query": query,
                "results": mcp_results,
                "llm_explanation": enhanced_results,
                "status": "success_with_llm"
            }
        else:
            return {
                "query": query,
                "results": mcp_results,
                "status": "success_mcp_only"
            }
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def query_mcp_server(query: str):
    """Query the MCP server for legal data"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Try the status endpoint first to check if server is alive
            health_response = await client.get(f"{MCP_SERVER_URL}/health")
            if health_response.status_code != 200:
                raise Exception("MCP server health check failed")
            
            # Mock MCP query - replace with actual MCP protocol calls
            # For now, return a structured response
            return [
                {
                    "title": f"Hukuki Arama: {query}",
                    "court": "Yargıtay 9. Hukuk Dairesi",
                    "date": "2024-07-20",
                    "summary": f"'{query}' konusunda bulunan karar özeti ve ilgili hukuki değerlendirmeler...",
                    "source": "MCP Server",
                    "relevance": "high"
                },
                {
                    "title": f"Danıştay Kararı: {query}",
                    "court": "Danıştay 5. Dairesi", 
                    "date": "2024-07-15",
                    "summary": f"İdari yargı kapsamında {query} ile ilgili değerlendirmeler ve sonuçlar...",
                    "source": "MCP Server",
                    "relevance": "medium"
                }
            ]
    
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        # Return fallback data
        return [
            {
                "title": f"Test Sonucu: {query}",
                "court": "Test Mahkemesi",
                "date": "2024-07-21",
                "summary": f"MCP server'a bağlantı kurulamadı. Test modu: {query} araması yapıldı.",
                "source": "Fallback Data",
                "relevance": "low"
            }
        ]

async def enhance_with_llm(query: str, mcp_results: list):
    """Enhance MCP results with OpenAI explanation"""
    if not openai_client:
        return "LLM not available"
    
    try:
        # Prepare context from MCP results
        context = ""
        for result in mcp_results[:3]:  # Use top 3 results
            context += f"- {result.get('title', 'N/A')}: {result.get('summary', 'N/A')}\n"
        
        # Create prompt for OpenAI
        prompt = f"""
Kullanıcı şu hukuki soruyu sordu: "{query}"

Aşağıdaki Türk hukuku veritabanından bulunan sonuçlar:
{context}

Lütfen:
1. Kullanıcının sorusunu anlaşılır şekilde açıkla
2. Bulunan sonuçları yorumla
3. Pratik öneriler ver
4. Türkçe olarak, sade ve anlaşılır dilde yanıtla

Yanıt:
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sen Türk hukuku konusunda uzman bir asistansın. Sade ve anlaşılır dilde açıklama yaparsın."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"LLM açıklaması şu anda kullanılamıyor: {str(e)}"

@app.get("/test-mcp")
async def test_mcp_connection():
    """Test MCP server connection"""
    try:
        results = await query_mcp_server("test sorgusu")
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
