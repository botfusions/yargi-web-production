from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
import asyncio
from typing import Optional, List, Dict, Any
import openai
import os
from datetime import datetime

# Import our fixed MCP client
 from backend.mcp_client import (
    search_legal_unified, 
    search_yargitay_only,
    get_mcp_tools,
    check_mcp_health,
    MCPClient
)
)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Yargı Web Interface",
    description="Turkish Legal Database Search with AI Analysis",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# OpenAI setup (optional)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    court_types: Optional[List[str]] = None
    daire: Optional[str] = None

class SimpleSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    query: str
    status: str
    llm_explanation: Optional[str] = None
    mcp_available: bool = True

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface"""
    # Check if LLM is available
    llm_enabled = bool(openai.api_key)
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "llm_enabled": llm_enabled
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check MCP server health
        mcp_health = await check_mcp_health()
        
        # Check tools availability
        tools = await get_mcp_tools()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "mcp_server": mcp_health.get("status", "unknown"),
            "tools_count": len(tools),
            "llm_enabled": bool(openai.api_key)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/tools")
async def get_tools():
    """Get available MCP tools"""
    try:
        tools = await get_mcp_tools()
        return {
            "tools": tools,
            "total": len(tools),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Failed to get tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_advanced(search_request: SearchRequest):
    """Advanced search endpoint"""
    try:
        logger.info(f"Advanced search: '{search_request.query}' with {search_request.limit} results")
        
        # Search using MCP
        if search_request.court_types and "yargitay" in search_request.court_types:
            # Use Yargıtay specific search
            result = await search_yargitay_only(
                search_request.query,
                search_request.limit,
                search_request.daire
            )
        else:
            # Use unified search
            result = await search_legal_unified(
                search_request.query,
                search_request.limit,
                search_request.court_types
            )
        
        # Format results
        if "error" in result:
            # Return fallback data
            return await get_fallback_response(search_request.query, search_request.limit)
        
        # Extract and format results
        formatted_results = await format_mcp_results(result, search_request.query)
        
        # Add LLM explanation if available
        llm_explanation = None
        if openai.api_key and formatted_results["results"]:
            try:
                llm_explanation = await get_llm_explanation(
                    search_request.query, 
                    formatted_results["results"][:2]  # Use first 2 results for explanation
                )
            except Exception as e:
                logger.error(f"LLM explanation failed: {e}")
        
        formatted_results["llm_explanation"] = llm_explanation
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Advanced search failed: {e}")
        # Return fallback response
        return await get_fallback_response(search_request.query, search_request.limit)

@app.post("/api/search-simple")
async def search_simple(query: str = Form(...)):
    """Simple search endpoint for frontend compatibility"""
    try:
        logger.info(f"Simple search: '{query}'")
        
        # Search using unified MCP search
        result = await search_legal_unified(query, limit=5)
        
        if "error" in result:
            logger.warning(f"MCP search error: {result['error']}")
            return await get_fallback_response(query, 5)
        
        # Format results
        formatted_results = await format_mcp_results(result, query)
        
        # Add LLM explanation if available
        llm_explanation = None
        if openai.api_key and formatted_results["results"]:
            try:
                llm_explanation = await get_llm_explanation(
                    query, 
                    formatted_results["results"][:2]
                )
                formatted_results["llm_explanation"] = llm_explanation
            except Exception as e:
                logger.error(f"LLM explanation failed: {e}")
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Simple search failed: {e}")
        return await get_fallback_response(query, 5)

async def format_mcp_results(mcp_result: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Format MCP results for frontend"""
    try:
        # Handle different MCP response formats
        if isinstance(mcp_result, dict):
            if "results" in mcp_result:
                results = mcp_result["results"]
            elif "decisions" in mcp_result:
                results = mcp_result["decisions"]
            elif "data" in mcp_result:
                results = mcp_result["data"]
            else:
                # Assume the whole dict is the results
                results = [mcp_result] if mcp_result else []
        elif isinstance(mcp_result, list):
            results = mcp_result
        else:
            results = []
        
        # Format each result
        formatted_results = []
        for item in results:
            if isinstance(item, dict):
                formatted_item = {
                    "title": item.get("title", item.get("baslik", "Başlık Bulunamadı")),
                    "court": item.get("court", item.get("mahkeme", item.get("daire", "Bilinmeyen Mahkeme"))),
                    "date": item.get("date", item.get("tarih", item.get("karar_tarihi", "Tarih Yok"))),
                    "summary": item.get("summary", item.get("ozet", item.get("icerik", "Özet mevcut değil"))),
                    "source": item.get("source", "Yargı MCP")
                }
                
                # Ensure summary is not too long
                if len(formatted_item["summary"]) > 300:
                    formatted_item["summary"] = formatted_item["summary"][:300] + "..."
                
                formatted_results.append(formatted_item)
        
        return {
            "results": formatted_results,
            "total": len(formatted_results),
            "query": query,
            "status": "real_mcp_results",
            "mcp_available": True
        }
        
    except Exception as e:
        logger.error(f"Failed to format MCP results: {e}")
        return {
            "results": [],
            "total": 0,
            "query": query,
            "status": "format_error",
            "mcp_available": False
        }

async def get_llm_explanation(query: str, results: List[Dict[str, Any]]) -> str:
    """Get LLM explanation for search results"""
    try:
        if not openai.api_key:
            return None
        
        # Prepare context from results
        context = ""
        for i, result in enumerate(results[:2], 1):
            context += f"{i}. {result['title']}\n"
            context += f"   Mahkeme: {result['court']}\n"
            context += f"   Tarih: {result['date']}\n"
            context += f"   Özet: {result['summary'][:200]}...\n\n"
        
        prompt = f"""Kullanıcı "{query}" konusunda hukuki arama yaptı. Aşağıdaki mahkeme kararları bulundu:

{context}

Bu sonuçlar hakkında kısa, anlaşılır ve profesyonel bir açıklama yaz. Hukuki terimler varsa açıkla. Türkçe yanıt ver."""

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sen bir hukuk asistanısın. Türk hukuku konularında açık ve anlaşılır açıklamalar yaparsın."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"LLM explanation failed: {e}")
        return None

async def get_fallback_response(query: str, limit: int) -> Dict[str, Any]:
    """Return fallback response when MCP is unavailable"""
    fallback_results = [
        {
            "title": "Yargıtay 9. Hukuk Dairesi 2024/1234 Sayılı Kararı",
            "court": "Yargıtay 9. Hukuk Dairesi",
            "date": "2024-07-15",
            "summary": f"'{query}' konusunda örnek karar. Bu fallback veridir - MCP server bağlantısı kurulamadığında gösterilir.",
            "source": "Fallback Data"
        },
        {
            "title": "Danıştay 5. Daire 2024/5678 Sayılı Kararı", 
            "court": "Danıştay 5. Daire",
            "date": "2024-07-10",
            "summary": f"'{query}' ile ilgili idari yargı kararı. Gerçek veriler için MCP server'ın aktif olması gerekir.",
            "source": "Fallback Data"
        }
    ]
    
    return {
        "results": fallback_results[:limit],
        "total": len(fallback_results[:limit]),
        "query": query,
        "status": "fallback_mode",
        "mcp_available": False
    }

# Test endpoint
@app.get("/test-mcp")
async def test_mcp():
    """Test MCP connection"""
    try:
        async with MCPClient() as client:
            # Test health
            health = await client.health_check()
            
            # Test tools
            tools = await client.get_available_tools()
            
            # Test search
            search_result = await client.search_bedesten_unified("test", limit=1)
            
            return {
                "status": "success",
                "health": health,
                "tools_count": len(tools),
                "search_test": search_result,
                "message": "MCP connection successful!"
            }
    except Exception as e:
        logger.error(f"MCP test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "MCP connection failed"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
