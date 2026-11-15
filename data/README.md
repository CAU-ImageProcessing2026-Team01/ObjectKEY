# Local data

개인 촬영 이미지와 등록 템플릿은 이 폴더에 보관합니다. 실제 데이터는 Git에서 제외됩니다.

평가 진입점은 현재 입력 확인용 UTF-8 텍스트 목록을 받습니다. 한 줄에 이미지 경로 하나를 적습니다.
상대 경로는 목록 파일이 있는 폴더 기준입니다. 공백이 있는 경로도 따옴표 없이 적습니다.
빈 줄과 `#`으로 시작하는 주석은 무시합니다. UTF-8 BOM과 CRLF도 허용합니다.

```text
# data/manifest.txt
reference.png
queries/same-object.jpg
queries/다른 물체.bmp
```

정답 라벨을 사용한 정확도 평가는 아직 구현하지 않았습니다.
