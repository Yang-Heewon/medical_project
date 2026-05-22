# Windows 로컬에서 repo 준비 후 서버에 올리는 방법

네 로컬 경로 기준:

```powershell
cd C:\Users\User\Downloads
Expand-Archive .\vision-rag-cxr-template.zip -DestinationPath .\
cd .\vision-rag-cxr
```

서버로 올리기:

```powershell
scp -r C:\Users\User\Downloads\vision-rag-cxr root@<서버IP>:/workspace/vision-rag-cxr
```

포트가 필요한 서버라면:

```powershell
scp -P <포트번호> -r C:\Users\User\Downloads\vision-rag-cxr root@<서버IP>:/workspace/vision-rag-cxr
```

서버 접속 후:

```bash
ssh -p <포트번호> root@<서버IP>
cd /workspace/vision-rag-cxr
docker build -f docker/Dockerfile -t vision-rag-cxr:latest .
docker compose -f docker/docker-compose.gpu.yaml run --rm vision-rag-cxr
```
