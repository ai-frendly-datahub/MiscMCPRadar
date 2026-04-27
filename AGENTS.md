# MiscMCPRadar

새로운 Standard Tier Radar를 시작할 때 기준이 되는 캐노니컬 템플릿 저장소입니다. 템플릿의 목적은 개별 Radar 저장소가 공통 파이프라인을 빠르게 복제·조정할 수 있게 하는 것이며, 워크스페이스 전체 현황 문서를 대신하지는 않습니다.

## PURPOSE

- RSS/API 수집 → 엔티티 분석 → DuckDB 저장 → HTML 리포트 생성의 표준 파이프라인 제공
- `main.py`, `config/`, `radar/`, `tests/`, `mcp_server/` 골격 제공
- 새 Radar 생성 시 복사본 출발점을 제공
- 공통 규칙은 `radar-core`와 맞추되, 템플릿 차원의 기본값과 예제를 유지

## STRUCTURE

```
Radar-Template/
├── main.py                         # 표준 collect -> analyze -> store -> report 파이프라인
├── radar/
│   ├── collector.py               # collect_sources() 래퍼/템플릿 구현
│   ├── analyzer.py                # article validation 보조
│   ├── reporter.py                # HTML report / index generation
│   ├── storage.py                 # RadarStorage 래퍼
│   ├── notifier.py                # Email/Webhook 알림
│   ├── date_storage.py            # dated snapshot / retention
│   ├── models.py                  # 템플릿용 도메인 모델
│   ├── config_loader.py           # YAML 로더
│   ├── common/                    # validator 등 공통 유틸
│   └── templates/                 # 기본 HTML 템플릿
├── radar_core/
│   ├── collector.py
│   ├── analyzer.py
│   ├── storage.py
│   └── models.py
│       # 템플릿 내부에 포함된 shared-style 모듈 복사본
├── config/
│   ├── config.yaml                # database_path, report_dir, raw_data_dir 등
│   └── categories/template.yaml   # 예시 category 정의
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

- 템플릿 변경은 기존 Radar 저장소에 자동 전파되지 않는다. 필요하면 후속 적용 대상을 별도로 정한다.
- `radar-core`와 중복되는 계약을 바꾸면 두 저장소 간 드리프트 여부를 확인한다.
- 새 Radar 출발점에 필요한 기본값을 유지하되, 특정 도메인 전용 로직은 템플릿에 넣지 않는다.
- `main.py`의 표준 파이프라인 형태는 가능한 한 유지한다.
- `config/categories/template.yaml`은 예시 파일이므로 구조 기준으로 삼고, 도메인명에 과적합시키지 않는다.

## KNOWN RISKS

- 저장소 내부 `radar_core/` 복사본은 별도 `radar-core` 저장소와 드리프트할 수 있다.
- 템플릿 문서에 워크스페이스 전체 저장소 목록을 박아 넣으면 금방 낡으므로 지양한다.

## COMMANDS

```bash
python main.py --category template --recent-days 7
pytest tests/ -v
python scripts/check_quality.py
```
