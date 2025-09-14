from fastapi import FastAPI
from .routers import battery

app = FastAPI(
    title       = "Energy API",
    description = "Comprehensive energy optimization and analysis API",
    version     = "1.0.0"
)

# Include routers
app.include_router(battery.router)

@app.get("/")
async def root():
    return {
        "message": "Energy API - Battery optimization and energy analysis",
        "endpoints": {
            "battery_optimization": "/battery/optimize",
            "docs"                : "/docs",
            "redoc"               : "/redoc"
        }
    }