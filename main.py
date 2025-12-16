import warnings
import uvicorn
from api.app_urban_renewal import app

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    uvicorn.run(app, port=int(8008), host='0.0.0.0')
