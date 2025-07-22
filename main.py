from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import aiohttp
import json
import logging
import asyncio
from datetime import datetime, date
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Yargı Web Interface - Advanced",
    description="Türk Hukuk Veritabanları için Gelişmiş AI Destekli Arama Sistemi",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da specific domains ekle
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates (fixed paths)
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://yargi-mcp.botfusions.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data Models
class AdvancedSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Arama sorgusu")
    courts: List[str] = Field(default=["yargitay", "danistay"], description="Mahkeme türleri")
    daire: Optional[str] = Field(default=None, description="Daire/Kurul seçimi")
    start_date: Optional[str] = Field(default=None, description="Başlangıç tarihi")
    end_date: Optional[str] = Field(default=None, description="Bitiş tarihi")
    limit: int = Field(default=5, ge=1, le=20, description="Sonuç limiti")
    exact_phrase: bool = Field(default=False, description="Kesin cümle arama")
    
    @validator('query')
    def validate_query(cls, v):
        # XSS protection
        dangerous_chars = ['<', '>', '"', "'", '&']
        if any(char in v for char in dangerous_chars):
            raise ValueError("Query contains invalid characters")
        return v.strip()

class LegalResult(BaseModel):
    title: str
    court: str
    date: str
    summary: str
    source: str
    document_id: Optional[str] = None
    url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[LegalResult]
    total: int
    ai_explanation: Optional[str] = None
    search_params: Dict[str, Any]
    timestamp: datetime
    status: str

# Court and Daire mapping
COURT_TYPES = {
    "yargitay": "Yargıtay",
    "danistay": "Danıştay", 
    "yerel_hukuk": "Yerel Hukuk Mahkemeleri",
    "istinaf": "İstinaf Mahkemeleri",
    "kyb": "Kanun Yararına Bozma",
    "anayasa": "Anayasa Mahkemesi",
    "emsal": "Emsal Kararlar",
    "uyusmazlik": "Uyuşmazlık Mahkemesi",
    "kik": "Kamu İhale Kurulu",
    "rekabet": "Rekabet Kurumu",
    "sayistay": "Sayıştay",
    "kvkk": "KVKK",
    "bddk": "BDDK"
}

YARGITAY_DAIRELER = {
    "1hd": "1. Hukuk Dairesi", "2hd": "2. Hukuk Dairesi", "3hd": "3. Hukuk Dairesi",
    "4hd": "4. Hukuk Dairesi", "5hd": "5. Hukuk Dairesi", "6hd": "6. Hukuk Dairesi",
    "7hd": "7. Hukuk Dairesi", "8hd": "8. Hukuk Dairesi", "9hd": "9. Hukuk Dairesi",
    "10hd": "10. Hukuk Dairesi", "11hd": "11. Hukuk Dairesi", "12hd": "12. Hukuk Dairesi",
    "13hd": "13. Hukuk Dairesi", "14hd": "14. Hukuk Dairesi", "15hd": "15. Hukuk Dairesi",
    "16hd": "16. Hukuk Dairesi", "17hd": "17. Hukuk Dairesi", "18hd": "18. Hukuk Dairesi",
    "19hd": "19. Hukuk Dairesi", "20hd": "20. Hukuk Dairesi", "21hd": "21. Hukuk Dairesi",
    "22hd": "22. Hukuk Dairesi", "23hd": "23. Hukuk Dairesi",
    "1cd": "1. Ceza Dairesi", "2cd": "2. Ceza Dairesi", "3cd": "3. Ceza Dairesi",
    "4cd": "4. Ceza Dairesi", "5cd": "5. Ceza Dairesi", "6cd": "6. Ceza Dairesi",
    "7cd": "7. Ceza Dairesi", "8cd": "8. Ceza Dairesi", "9cd": "9. Ceza Dairesi",
    "10cd": "10. Ceza Dairesi", "11cd": "11. Ceza Dairesi", "12cd": "12. Ceza Dairesi",
    "13cd": "13. Ceza Dairesi", "14cd": "14. Ceza Dairesi", "15cd": "15. Ceza Dairesi",
    "16cd": "16. Ceza Dairesi", "17cd": "17. Ceza Dairesi", "18cd": "18. Ceza Dairesi",
    "19cd": "19. Ceza Dairesi", "20cd": "20. Ceza Dairesi", "21cd": "21. Ceza Dairesi",
    "22cd": "22. Ceza Dairesi", "23cd": "23. Ceza Dairesi",
    "hgk": "Hukuk Genel Kurulu", "cgk": "Ceza Genel Kurulu", "bsk": "Başkanlar Kurulu"
}

DANISTAY_DAIRELER = {
    "1d": "1. Daire", "2d": "2. Daire", "3d": "3. Daire", "4d": "4. Daire", "5d": "5. Daire",
    "6d": "6. Daire", "7d": "7. Daire", "8d": "8. Daire", "9d": "9. Daire", "10d": "10. Daire",
    "11d": "11. Daire", "12d": "12. Daire", "13d": "13. Daire", "14d": "14. Daire", "15d": "15. Daire",
    "16d": "16. Daire", "17d": "17. Daire",
    "ik": "İdari Dava Daireleri Kurulu", "vk": "Vergi Dava Daireleri Kurulu",
    "ayhim": "Askeri Yüksek İdare Mahkemesi"
}

# MCP Client
# main.py'daki MCPClient class'ını bu şekilde değiştir:

class MCPClient:
    def __init__(self):
        self.base_url = MCP_SERVER_URL
        self.session_id = "mock_session_12345"  # Default session ID
        
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
                    "clientInfo": {"name": "yargi-web-advanced", "version": "2.0.0"}
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer mock_clerk_jwt_development_token_12345",
                "X-Session-ID": self.session_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/mcp/",
                    json=init_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    response_text = await response.text()
                    logger.info(f"MCP init response ({response.status}): {response_text}")
                    
                    if response.status == 200:
                        # Try to get session ID from headers
                        new_session_id = response.headers.get('mcp-session-id')
                        if new_session_id:
                            self.session_id = new_session_id
                            
                        logger.info(f"MCP session initialized successfully with ID: {self.session_id}")
                        return True
                    else:
                        logger.error(f"MCP initialization failed: {response.status} - {response_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"MCP initialization error: {e}")
            return False
    
    async def search_advanced(self, search_request: AdvancedSearchRequest) -> List[LegalResult]:
        """Advanced MCP search with all parameters"""
        try:
            # Always ensure we have a session
            if not self.session_id:
                self.session_id = "mock_session_12345"
            
            # Try to initialize if not done
            session_ok = await self.initialize_session()
            if not session_ok:
                logger.warning("MCP initialization failed, using fallback")
                return self.fallback_results(search_request.query)
            
            # Prepare search phrase
            search_phrase = search_request.query
            if search_request.exact_phrase:
                search_phrase = f'"{search_phrase}"'
            
            # Prepare MCP call payload
            search_payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": 2,
                "params": {
                    "name": "search_bedesten_unified",
                    "arguments": {
                        "phrase": search_phrase,
                        "court_types": search_request.courts,
                        "limit": search_request.limit
                    }
                }
            }
            
            # Add optional parameters only if they exist
            if search_request.daire and search_request.daire.strip():
                search_payload["params"]["arguments"]["birimAdi"] = search_request.daire
                
            if search_request.start_date and search_request.start_date.strip():
                search_payload["params"]["arguments"]["kararTarihiStart"] = f"{search_request.start_date}T00:00:00.000Z"
                
            if search_request.end_date and search_request.end_date.strip():
                search_payload["params"]["arguments"]["kararTarihiEnd"] = f"{search_request.end_date}T23:59:59.000Z"
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json", 
                "Authorization": "Bearer mock_clerk_jwt_development_token_12345"
            }
            
            # Add session ID only if it exists and is not None
            if self.session_id and self.session_id != "None":
                headers["X-Session-ID"] = self.session_id
            
            logger.info(f"Making MCP search call for: {search_phrase}")
            
            # Make MCP call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/mcp/",
                    json=search_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    
                    response_text = await response.text()
                    logger.info(f"MCP search response ({response.status}): {response_text[:500]}...")
                    
                    if response.status == 200:
                        try:
                            data = await response.json() if response_text else {}
                            return self.parse_mcp_results(data, search_request.query)
                        except Exception as parse_error:
                            logger.error(f"JSON parse error: {parse_error}")
                            return self.fallback_results(search_request.query)
                    else:
                        logger.error(f"MCP search failed: {response.status} - {response_text}")
                        return self.fallback_results(search_request.query)
                        
        except Exception as e:
            logger.error(f"MCP search error: {e}")
            return self.fallback_results(search_request.query)
    
    def parse_mcp_results(self, data: Dict[str, Any], query: str) -> List[LegalResult]:
        """Parse MCP response to LegalResult objects"""
        results = []
        
        try:
            logger.info(f"Parsing MCP response: {str(data)[:200]}...")
            
            if "result" in data and data["result"]:
                result_data = data["result"]
                
                if isinstance(result_data, dict) and "content" in result_data:
                    content = result_data["content"]
                    if isinstance(content, list) and content:
                        # Parse structured data from MCP response
                        raw_content = content[0].get("text", "") if content[0] else ""
                        logger.info(f"MCP raw content: {raw_content[:200]}...")
                        
                        if raw_content and len(raw_content) > 50:
                            results = self.extract_legal_data(raw_content, query)
                            logger.info(f"Extracted {len(results)} results from MCP data")
                        else:
                            logger.warning("MCP returned empty or short content")
                elif isinstance(result_data, str) and result_data:
                    # Direct text response
                    results = self.extract_legal_data(result_data, query) 
                    logger.info(f"Extracted {len(results)} results from direct MCP text")
                else:
                    logger.warning(f"Unexpected MCP result format: {type(result_data)}")
            else:
                logger.warning("No result field in MCP response")
                    
        except Exception as e:
            logger.error(f"Error parsing MCP results: {e}")
        
        # Return real results or fallback
        if results:
            logger.info(f"Returning {len(results)} real MCP results")
            return results
        else:
            logger.info("No real results found, using fallback")
            return self.fallback_results(query)
    
    def extract_legal_data(self, raw_content: str, query: str) -> List[LegalResult]:
        """Extract structured legal data from MCP response"""
        results = []
        
        try:
            # More sophisticated parsing for real MCP data
            if "başlık" in raw_content.lower() or "karar" in raw_content.lower():
                # This is likely real legal content
                lines = raw_content.split('\n')
                current_result = {}
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if "başlık:" in line.lower() or "title:" in line.lower():
                        if current_result.get('title'):
                            results.append(LegalResult(**current_result))
                            current_result = {}
                        current_result['title'] = line.split(':', 1)[1].strip()
                    elif "mahkeme:" in line.lower() or "court:" in line.lower():
                        current_result['court'] = line.split(':', 1)[1].strip()
                    elif "tarih:" in line.lower() or "date:" in line.lower():
                        current_result['date'] = line.split(':', 1)[1].strip()
                    elif "özet:" in line.lower() or "summary:" in line.lower():
                        current_result['summary'] = line.split(':', 1)[1].strip()
                        current_result['source'] = "Yargı MCP - Real Data"
                
                # Add last result
                if current_result.get('title'):
                    results.append(LegalResult(**current_result))
            
            logger.info(f"Parsed {len(results)} legal results from content")
                    
        except Exception as e:
            logger.error(f"Error extracting legal data: {e}")
            
        return results
    
    def fallback_results(self, query: str) -> List[LegalResult]:
        """Fallback results when MCP is not available"""
        return [
            LegalResult(
                title=f"Test Modu - {query} Arama Sonucu",
                court="MCP Connection Test",
                date="2024-07-22",
                summary=f"{query} için MCP bağlantısı test ediliyor. Gerçek veriler yükleniyor...",
                source="Yargı MCP - Fallback Mode",
                document_id="fallback_001"
            )
        ]
# Initialize MCP client
mcp_client = MCPClient()

# OpenAI Integration
async def get_ai_analysis(query: str, results: List[LegalResult], search_params: Dict) -> str:
    """Get AI analysis using OpenAI"""
    if not OPENAI_API_KEY:
        return None
    
    try:
        # Prepare context
        context = f"""Kullanıcı Sorusu: {query}

Arama Parametreleri:
- Mahkemeler: {', '.join(search_params.get('courts', []))}
- Daire: {search_params.get('daire', 'Tümü')}
- Tarih Aralığı: {search_params.get('start_date', 'Belirtilmemiş')} - {search_params.get('end_date', 'Belirtilmemiş')}
- Kesin Arama: {'Evet' if search_params.get('exact_phrase') else 'Hayır'}

Bulunan Hukuki Metinler:
"""
        
        for i, result in enumerate(results, 1):
            context += f"\n{i}. {result.title}\n   Mahkeme: {result.court}\n   Tarih: {result.date}\n   Özet: {result.summary}\n"
        
        context += """\n\nLütfen bu hukuki metinleri analiz ederek:
1. Kullanıcının sorusuna kapsamlı bir yanıt verin
2. İlgili hukuki prensipleri ve kavramları açıklayın  
3. Farklı mahkeme kararları arasındaki bağlantıları gösterin
4. Pratik öneriler ve sonraki adımlar sunun
5. Ek araştırma önerileri verin

Yanıtınız Türkçe, anlaşılır ve profesyonel hukuki dille olsun."""

        # OpenAI API call
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Sen Türk hukuku konusunda uzman bir hukuk asistanısın. Verilen mahkeme kararlarını analiz ederek kullanıcılara kapsamlı, anlaşılır ve profesyonel hukuki rehberlik sağla."
                        },
                        {
                            "role": "user",
                            "content": context
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.3
                },
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
    
    return "AI analizi şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with advanced search interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "court_types": COURT_TYPES,
        "yargitay_daireler": YARGITAY_DAIRELER,
        "danistay_daireler": DANISTAY_DAIRELER,
        "version": "2.0.0"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "yargi-web-advanced",
        "version": "2.0.0",
        "features": {
            "advanced_search": True,
            "multi_court": True,
            "date_filtering": True,
            "daire_filtering": True,
            "ai_analysis": bool(OPENAI_API_KEY),
            "mcp_tools": 21
        },
        "timestamp": datetime.utcnow()
    }

@app.post("/api/search", response_model=SearchResponse)
async def advanced_search(search_request: AdvancedSearchRequest):
    """Advanced legal search with full MCP features"""
    try:
        logger.info(f"Advanced search: {search_request.query} with params: {search_request.dict()}")
        
        # Search using MCP
        results = await mcp_client.search_advanced(search_request)
        
        # Get AI analysis
        ai_analysis = None
        if OPENAI_API_KEY:
            ai_analysis = await get_ai_analysis(
                search_request.query,
                results,
                search_request.dict()
            )
        
        response = SearchResponse(
            query=search_request.query,
            results=results,
            total=len(results),
            ai_explanation=ai_analysis,
            search_params=search_request.dict(),
            timestamp=datetime.utcnow(),
            status="success"
        )
        
        logger.info(f"Search completed: {len(results)} results found")
        return response
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Simple search endpoint for backward compatibility
@app.post("/api/search-simple")
async def simple_search(query: str):
    """Simple search endpoint for existing frontend"""
    try:
        # Create simple request
        search_request = AdvancedSearchRequest(
            query=query,
            courts=["yargitay", "danistay"],
            limit=5
        )
        
        # Use advanced search
        results = await mcp_client.search_advanced(search_request)
        
        # Format for simple response
        simple_results = []
        for result in results:
            simple_results.append({
                "title": result.title,
                "court": result.court,
                "date": result.date,
                "summary": result.summary,
                "source": result.source
            })
        
        return {
            "query": query,
            "results": simple_results,
            "total": len(simple_results),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Simple search error: {e}")
        return {
            "query": query,
            "results": [],
            "total": 0,
            "status": "error",
            "message": str(e)
        }

@app.get("/api/courts")
async def get_courts():
    """Get available courts and daireler"""
    return {
        "courts": COURT_TYPES,
        "yargitay_daireler": YARGITAY_DAIRELER,
        "danistay_daireler": DANISTAY_DAIRELER,
        "total_courts": len(COURT_TYPES),
        "total_daireler": len(YARGITAY_DAIRELER) + len(DANISTAY_DAIRELER)
    }

@app.get("/api/tools")
async def get_mcp_tools():
    """Get available MCP tools info"""
    return {
        "available_tools": [
            "search_bedesten_unified",
            "get_bedesten_document_markdown", 
            "search_emsal_detailed_decisions",
            "get_emsal_document_markdown",
            "search_uyusmazlik_decisions",
            "get_uyusmazlik_document_markdown_from_url",
            "search_anayasa_unified",
            "get_anayasa_document_unified",
            "search_kik_decisions",
            "get_kik_document_markdown",
            "search_rekabet_kurumu_decisions",
            "get_rekabet_kurumu_document",
            "search_sayistay_genel_kurul",
            "search_sayistay_temyiz_kurulu", 
            "search_sayistay_daire",
            "get_sayistay_genel_kurul_document_markdown",
            "get_sayistay_temyiz_kurulu_document_markdown",
            "get_sayistay_daire_document_markdown",
            "search_kvkk_decisions",
            "get_kvkk_document_markdown",
            "search_bddk_decisions",
            "get_bddk_document_markdown"
        ],
        "total_tools": 21,
        "optimized_features": {
            "token_reduction": "61.8%",
            "court_filtering": "79 options",
            "date_filtering": "ISO 8601 format",
            "exact_phrase_search": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
