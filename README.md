# MiscMCPRadar - 기타 MCP 레이더

**🌐 Live Report**: https://ai-frendly-datahub.github.io/MiscMCPRadar/


Miscellaneous 섹션의 한국 MCP 서버 목록을 수집하고 리스크/활성도를 추적하는 독립 Radar입니다.

## 프로젝트 목표

- **데이터 수집**: RSS 피드 및 API
- **엔티티 분석**: 도메인별 키워드 매칭
- **트렌드 리포트**: DuckDB 저장 + HTML 리포트로 {domain} 동향 시각화
- **자동화**: GitHub Actions 일일 수집 + GitHub Pages 리포트 자동 배포

## 기술적 우수성

- **안정성**: HTTP 자동 재시도(지수 백오프), DB 트랜잭션 에러 처리
- **관찰성**: 구조화된 JSON 로깅으로 파이프라인 상태 실시간 모니터링
- **품질 보증**: 단위 테스트로 코드 변경 시 회귀 버그 사전 차단
- **고성능**: 배치 처리 최적화로 대량 데이터 수집 시 성능 향상
- **운영 자동화**: Email/Webhook 알림으로 무인 운영 가능

## 빠른 시작

1. 가상환경을 만들고 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

2. 실행:
   ```bash
   python main.py --category misc_mcp --recent-days 7
   # 리포트: reports/misc_mcp_report.html
   ```

   주요 옵션: `--per-source-limit 20`, `--recent-days 5`, `--keep-days 60`, `--timeout 20`.

## GitHub Actions & GitHub Pages

- 워크플로: `.github/workflows/radar-crawler.yml`
  - 스케줄: 매일 00:00 UTC (KST 09:00), 수동 실행도 지원.
  - 환경 변수 `RADAR_CATEGORY`를 프로젝트에 맞게 수정하세요.
  - 리포트 배포 디렉터리: `reports` → `gh-pages` 브랜치로 배포.
  - DuckDB 경로: `data/radar_data.duckdb` (Pages에 올라가지 않음). 아티팩트로 7일 보관.

- 설정 방법:
  1) 저장소 Settings → Pages에서 `gh-pages` 브랜치를 선택해 활성화
  2) Actions 권한을 기본값으로 두거나 외부 PR에서도 실행되도록 설정
  3) 워크플로 파일의 `RADAR_CATEGORY`를 원하는 YAML 이름으로 변경

## 동작 방식

- **수집**: 카테고리 YAML에 정의된 소스를 수집합니다. 실행 시 DuckDB에 적재하고 보존 기간(`keep_days`)을 적용합니다.
- **분석**: 엔티티별 키워드 매칭. 매칭된 키워드를 리포트에 칩으로 표시합니다.
- **리포트**: `reports/<category>_report.html`을 생성하며, 최근 N일(기본 7일) 기사와 엔티티 히트 카운트, 수집 오류를 표시합니다.

## 기본 경로

- DB: `data/radar_data.duckdb`
- 리포트 출력: `reports/`

## 디렉터리 구성

```
Radar-Template/
  main.py                 # CLI 엔트리포인트
  requirements.txt        # 의존성
  config/
    config.yaml           # DB/리포트 경로 설정
    categories/
      misc_mcp.yaml  # 소스 + 엔티티 정의
  radar/
    collector.py          # 데이터 수집
    analyzer.py           # 엔티티 태깅
    reporter.py           # HTML 렌더링
    storage.py            # DuckDB 저장/정리
    config_loader.py      # YAML 로더
    models.py             # 데이터 클래스
  .github/workflows/      # GitHub Actions (crawler + Pages 배포)
```
## MCP 운영 범위

- Seed source: `https://github.com/darjeeling/awesome-mcp-korea`
- Source section: `Miscellaneous`
- Primary motion: `intelligence`
- Governance: `medium`
- Data quality priority: `P3`

이 Radar는 awesome list를 `T4_community` seed source로만 사용합니다. 각 MCP 서버의 실제 보안성, credential 요구, write action 여부는 linked GitHub repository metadata와 README를 별도로 검증해야 합니다.
