from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint to ensure the API is running.
    
    Returns:
        dict: A simple status message.
    """
    return {"status": "OK"}