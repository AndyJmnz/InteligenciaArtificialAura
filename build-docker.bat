@echo off
echo [1/4] Limpiando contenedores e imagenes anteriores...
docker stop aura-api 2>nul
docker rm aura-api 2>nul
docker rmi aura-ai 2>nul

echo [2/4] Construyendo imagen Docker...
docker build -t aura-ai . --no-cache

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Fallo la construccion de la imagen
    pause
    exit /b 1
)

echo [3/4] Ejecutando contenedor...
docker run -d -p 8000:8000 --name aura-api aura-ai

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Fallo la ejecucion del contenedor
    pause
    exit /b 1
)

echo [4/4] Verificando estado...
timeout /t 5
docker ps | findstr aura-api

echo.
echo ✅ API disponible en: http://localhost:8000
echo ✅ Documentacion en: http://localhost:8000/docs
echo ✅ Health check: http://localhost:8000/health
echo.
echo Para ver logs: docker logs -f aura-api
echo Para detener: docker stop aura-api
echo.
pause