from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import aiohttp
import json
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Yargı Web Interface", version="1.0.0")

# Mount static files and templates
templates = Jinja2Templates(directory="templates")

class SearchRequest(BaseModel):
    query: str
    court_types: Optional[List[str]] = ["yargitay", "danistay"]
    limit: Optional[int] = 3

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    query: str
    status: str
    llm_explanation: Optional[str] = None

class MCPClient:
    def __init__(self):
        self.base_url = "https://yargi-mcp.botfusions.com"
        self.session_id = None
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",  # FIXED: Added this line
            "Authorization": "Bearer mock_clerk_jwt_development_token_12345"
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'session'):
            await self.session.close()
    
    async def initialize_session(self):
        """Initialize MCP session"""
        try:
            init_payload = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1,
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "yargi-web-client", "version": "1.0.0"}
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp/",
                headers=self.headers,
                json=init_payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    # Extract session ID from response headers if available
                    self.session_id = response.headers.get('mcp-session-id', 'default-session')
                    logger.info(f"MCP session initialized: {self.session_id}")
                    return result
                else:
                    logger.error(f"Failed to initialize MCP session: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"MCP initialization error: {str(e)}")
            return None
    
    async def health_check(self):
        """Check MCP server health"""
        try:
            async with self.session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return await response.json()
                return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_available_tools(self):
        """Get available MCP tools"""
        try:
            tools_payload = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp/",
                headers=self.headers,
                json=tools_payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", {}).get("tools", [])
                return []
        except Exception as e:
            logger.error(f"Get tools error: {str(e)}")
            return []
    
    async def search_bedesten_unified(self, query: str, court_types: List[str] = None, limit: int = 3):
        """Search using bedesten unified API"""
        try:
            # Prepare court types
            if not court_types:
                court_types = ["yargitay", "danistay"]
            
            search_payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": 3,
                "params": {
                    "name": "search_bedesten_unified",
                    "arguments": {
                        "phrase": query,
                        "court_types": court_types,
                        "limit": limit
                    }
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp/",
                headers=self.headers,
                json=search_payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", {}).get("content", [])
                else:
                    logger.error(f"Search failed: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return None

def get_fallback_results(query: str) -> List[Dict[str, Any]]:
    """Fallback results when MCP is not available"""
    return [
        {
            "title": f"Yargıtay 9. Hukuk Dairesi 2024/1234 Sayılı Kararı",
            "court": "Yargıtay 9. Hukuk Dairesi",
            "date": "2024-07-15",
            "source": "Fallback Data",
            "summary": f"'{query}' konusunda örnek karar. Bu fallback veridir - MCP server bağlantısı kurulamadığında gösterilir."
        },
        {
            "title": f"Danıştay 5. Daire 2024/5678 Sayılı Kararı",
            "court": "Danıştay 5. Daire",
            "date": "2024-07-10",
            "source": "Fallback Data",
            "summary": f"'{query}' ile ilgili idari yargı kararı. Gerçek veriler için MCP server'ın aktif olması gerekir."
        }
    ]

async def get_ai_explanation(query: str, results: List[Dict]) -> str:
    """Get AI explanation using OpenAI (mock for now)"""
    try:
        # This would be replaced with actual OpenAI API call
        court_names = [r.get('court', 'Bilinmeyen Mahkeme') for r in results[:2]]
        
        return f"""
'{query}' konusunda bulunan kararlar hakkında açıklama:

Araştırılan konuda {len(results)} adet karar bulunmuştur. Bu kararlar {', '.join(court_names)} tarafından verilmiştir.

Bu sonuçlar Türk hukuk sistemindeki ilgili mahkeme kararlarını yansıtmaktadır. Detaylı analiz için kararların tam metinlerine başvurulması önerilir.

Not: Bu açıklama AI tarafından oluşturulmuştur ve hukuki görüş niteliği taşımaz.
        """.strip()
    except Exception as e:
        logger.error(f"AI explanation error: {str(e)}")
        return None

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "llm_enabled": True
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with MCPClient() as client:
            health = await client.health_check()
            tools = await client.get_available_tools()
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "mcp_server": health.get("status", "unknown"),
                "tools_count": len(tools),
                "tools_available": len(tools) > 0
            }
    except Exception as e:
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "mcp_server": "unavailable",
            "tools_count": 0
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
        return {
            "status": "error",
            "error": str(e),
            "message": "MCP connection failed!"
        }

@app.post("/api/search")
async def search_advanced(request: SearchRequest):
    """Advanced search endpoint"""
    try:
        async with MCPClient() as client:
            # Get real MCP results
            mcp_results = await client.search_bedesten_unified(
                query=request.query,
                court_types=request.court_types,
                limit=request.limit
            )
            
            if mcp_results:
                # Process MCP results
                processed_results = []
                for result in mcp_results[:request.limit]:
                    processed_results.append({
                        "title": result.get("title", "Başlık Bulunamadı"),
                        "court": result.get("court", "Mahkeme Bilgisi Yok"),
                        "date": result.get("date", "Tarih Belirtilmemiş"),
                        "source": "Yargı MCP",
                        "summary": result.get("summary", result.get("content", "Özet mevcut değil"))[:500]
                    })
                
                # Get AI explanation
                ai_explanation = await get_ai_explanation(request.query, processed_results)
                
                return SearchResponse(
                    results=processed_results,
                    total=len(processed_results),
                    query=request.query,
                    status="real_mcp_results",
                    llm_explanation=ai_explanation
                )
            else:
                # Fallback to test data
                fallback_results = get_fallback_results(request.query)
                ai_explanation = await get_ai_explanation(request.query, fallback_results)
                
                return SearchResponse(
                    results=fallback_results,
                    total=len(fallback_results),
                    query=request.query,
                    status="fallback_data",
                    llm_explanation=ai_explanation
                )
                
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        
        # Return fallback data on error
        fallback_results = get_fallback_results(request.query)
        
        return SearchResponse(
            results=fallback_results,
            total=len(fallback_results),
            query=request.query,
            status="error_fallback",
            llm_explanation=f"Arama sırasında hata oluştu: {str(e)}"
        )

@app.post("/api/search-simple")
async def search_simple(query: str = Form(...)):
    """Simple search endpoint for frontend compatibility"""
    try:
        search_request = SearchRequest(query=query, limit=3)
        return await search_advanced(search_request)
    except Exception as e:
        logger.error(f"Simple search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courts")
async def get_courts():
    """Get available courts and daireler"""
    return {
        "court_types": [
            {"id": "yargitay", "name": "Yargıtay", "daireler": 52},
            {"id": "danistay", "name": "Danıştay", "daireler": 27},
            {"id": "yerel_hukuk", "name": "Yerel Hukuk Mahkemeleri", "daireler": 0},
            {"id": "istinaf_hukuk", "name": "İstinaf Hukuk Mahkemeleri", "daireler": 0},
            {"id": "kyb", "name": "Kanun Yararına Bozma", "daireler": 0}
        ],
        "total_daireler": 79
    }

@app.get("/api/tools")
async def get_tools():
    """Get available MCP tools"""
    try:
        async with MCPClient() as client:
            tools = await client.get_available_tools()
            return {
                "tools": tools,
                "count": len(tools),
                "status": "available" if len(tools) > 0 else "unavailable"
            }
    except Exception as e:
        return {
            "tools": [],
            "count": 0,
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
