# MiscMCPRadar

MiscMCPRadar는 기타 MCP 후보를 수집하고 repository metadata, capability, risk signal을 DuckDB/HTML 리포트로 정리하는 MCP Radar 저장소입니다.

## PURPOSE

- RSS/API 수집 → 엔티티 분석 → DuckDB 저장 → HTML 리포트 생성의 표준 파이프라인 제공
- `main.py`, `config/`, `radar/`, `tests/`, `mcp_server/` 골격 제공
- MCP 후보의 capability discovery, risk triage, activation gate 점검 산출물 제공
- 공통 규칙은 `radar-core`, MCP source contract, repo-local category YAML에 맞춘다

## STRUCTURE

```
MiscMCPRadar/
├── main.py                         # 표준 collect -> analyze -> store -> report 파이프라인
├── radar/
│   ├── collector.py               # collect_sources() 래퍼/MCP 후보 수집 구현
│   ├── analyzer.py                # article validation 보조
│   ├── reporter.py                # HTML report / index generation
│   ├── storage.py                 # RadarStorage 래퍼
│   ├── notifier.py                # Email/Webhook 알림
│   ├── date_storage.py            # dated snapshot / retention
│   ├── models.py                  # MCP Radar 도메인 모델
│   ├── config_loader.py           # YAML 로더
│   ├── common/                    # validator 등 공통 유틸
│   └── templates/                 # 기본 HTML 템플릿
├── radar_core/
│   ├── collector.py
│   ├── analyzer.py
│   ├── storage.py
│   └── models.py
│       # repo-local shared-style 모듈 복사본
├── config/
│   ├── config.yaml                # database_path, report_dir, raw_data_dir 등
│   └── categories/misc_mcp.yaml   # MCP category 정의
├── mcp_server/                    # MCP server/tools 골격
├── scripts/check_quality.py       # 품질 검사 스크립트
├── tests/
│   ├── unit/
│   └── integration/
├── reports/                       # 생성 결과 예시
└── .github/workflows/             # crawler / pages 배포 워크플로
```

## PIPELINE

```text
main.py
  -> load_settings()
  -> load_category_config()
  -> collect_sources()
  -> RawLogger.log()
  -> apply_entity_rules()
  -> validate_article()
  -> RadarStorage.upsert_articles()
  -> SearchIndex.upsert()
  -> generate_report()
  -> generate_index_html()
  -> apply_date_storage_policy()
  -> optional notifications
```

## MODIFICATION RULES

- MCP Radar 공통 계약 변경은 다른 MCP 저장소에 자동 전파되지 않는다. 필요하면 후속 적용 대상을 별도로 정한다.
- `radar-core`와 중복되는 계약을 바꾸면 두 저장소 간 드리프트 여부를 확인한다.
- MCP 후보별 metadata와 activation gate는 category YAML과 테스트에 함께 반영한다.
- `main.py`의 표준 파이프라인 형태는 가능한 한 유지한다.
- `config/categories/misc_mcp.yaml`은 MCP 후보 metadata, source contract, activation gate의 기준 파일로 유지한다.

## KNOWN RISKS

- 저장소 내부 `radar_core/` 복사본은 별도 `radar-core` 저장소와 드리프트할 수 있다.
- MCP 후보 metadata, activation gate, runtime evidence가 category YAML과 README에서 엇갈리지 않게 유지한다.

## COMMANDS

```bash
python main.py --category misc_mcp --recent-days 7
pytest tests/ -v
python scripts/check_quality.py
```
