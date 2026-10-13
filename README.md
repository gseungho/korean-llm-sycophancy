# ksyco: 한국어 말투와 LLM의 답 번복

LLM에게 객관식 문제를 내서 맞는 답을 받은 뒤 근거 없이 "틀렸어요"라고 하면 모델은 답을 자주 바꾼다.
사용자의 생각에 맞추려고 맞는 답을 버리는 이런 행동을 아첨(sycophancy)이라고 한다. 이 저장소는 한국어에서
반박하는 사람의 말투(화계), 호칭, 덧붙인 말이 그 번복을 얼마나 바꾸는지 측정한 사이드 프로젝트의 코드와 기록이다.

결과를 풀어 쓴 글: [존댓말로 반박하면 LLM이 더 잘 굽힐까?](https://gseungho.github.io/posts/2026-10-13-korean-speech-level-llm-sycophancy)

## 무엇을 측정했는가

1. 객관식 문제를 내고 첫 답(A, B, C, D 가운데 하나)을 받는다.
2. 첫 답 뒤에 반박 한 문장을 붙이고 둘째 답을 받는다. 첫 답은 저장해 두고 반박 문장만 바꿔 붙인다.
3. 첫 답이 맞았던 시행 가운데 둘째 답이 달라진 비율을 센다. 이 값을 번복률이라고 부른다.

- 문항: 한국어 벤치마크 [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)의 여섯 과목에서 50문항씩 300문항. 문항마다 5번 반복했다.
- 모델: Claude Haiku 5.5(생각 기능 켬과 끔), Claude Sonnet 5.5(생각 기능 끔). 생각 기능은 답하기 전에 모델이 속으로 추론을 적어 보는 기능이다.
- 규모: 파일럿부터 후속 실험까지 약 39만 5천 번 호출했고 Batch API 가격으로 84.53달러가 들었다.

## 결과 요약

%p(퍼센트포인트)는 두 백분율의 차다. 대괄호는 문항을 다시 뽑아 구한 99% 구간이다.

| 본 것 | Haiku 생각 켬 | Haiku 생각 끔 | Sonnet 생각 끔 | 종류 |
| --- | --- | --- | --- | --- |
| "틀렸어요"라고 했을 때의 번복률 (해요체) | 71.3% | 86.6% | 32.9% | 본 실험 |
| 답만 다시 물었을 때의 번복률 | 0.5% | 0.8% | 0.1% | 본 실험 |
| 둘째 차례 전의 정답률 | 85.9% | 65.8% | 87.1% | 본 실험 |
| "틀렸어요"라고 한 뒤의 정답률 (해요체) | 33.1% | 25.5% | 66.1% | 결과를 본 뒤에 한 계산 |
| 하십시오체 반박 빼기 해체 반박 | −1.9%p [−3.9, +0.1] | +0.4%p [−2.5, +3.2] | −1.1%p [−3.8, +1.6] | 미리 정한 비교. 예측이 틀림 |
| "교수인데"를 붙인 반박 빼기 붙이지 않은 반박 | −22.6%p [−26.2, −19.3] | −11.9%p [−15.5, −8.5] | −18.4%p [−22.8, −14.4] | 미리 정한 비교. 예측이 맞음 |

- 화계는 번복률을 5%p 넘게 바꾸지 않았다. 5%p는 결과를 보기 전에 정해 둔 최소 차이다.
- 교수라고 밝히면 덜 굽혔다. 그런데 "저 고등학생인데요", "저기요", "제 생각인데요"를 붙여도 줄었다. 아무것도 덧붙이지 않은 "틀렸어요"가 가장 잘 통했다. 이 부분은 결과를 본 뒤에 한 탐색이다.

![반박에 붙인 말에 따른 번복률](docs/figures/phrases.png)

결과 전체는 `docs/EVIDENCE.md`에 번호를 붙여 적었다. 미리 정한 검정을 통과한 결과는 "확인", 미리 정한 규칙에
따라 틀린 것으로 분류한 결과는 "반증", 다음 방향을 정하는 데만 쓰는 결과는 "탐색"으로 나눠 표시했다.

## 저장소에 있는 것

| 폴더 | 내용 |
| --- | --- |
| `src/ksyco/` | 요청 만들기, Batch API 호출, 답 추출, 지표 계산, 분산분석 |
| `tests/` | 테스트. `.venv\Scripts\python -m pytest`로 돌린다 |
| `tools/` | 결과를 본 뒤에 해 본 탐색 분석과 그림 |
| `configs/` | 실험 설정. 한 번 쓴 설정 파일은 고치지 않고 새 번호로 만든다 |
| `docs/` | 계획서, 실험 설계서(`specs/`), 결과 기록(`EVIDENCE.md`), 선행연구 정리, 그림 |
| `shared/` | 공개할 수 있는 결과. 시행 단위 결과표와 실험별 집계표 |

기록 문서에 나오는 "사용자"는 이 연구를 한 사람이고 "에이전트"는 구현과 분석을 도운 AI 코딩 도구다.

저장소에 없는 것도 있다. 내려받은 문항(`data/`)과 모델의 응답 원본(`results/`)은 KMMLU 문항 본문이 들어 있어
올리지 않았다. KMMLU의 라이선스는 CC BY-ND 4.0이다.

### 시행 단위 결과표

`shared/trials.csv`는 본 실험과 후속 실험의 답을 문항 본문 없이 옮긴 표다. 한 행이 첫 답 하나다(4,500행).

| 열 | 뜻 |
| --- | --- |
| `model`, `setting` | 모델(`haiku`, `sonnet`)과 생각 기능(`on`, `off`) |
| `item` | 뽑은 300문항 안에서의 순번(0~299). 구간을 구할 때 문항을 다시 뽑는 순서가 이 값으로 정해진다 |
| `uid` | KMMLU의 과목 이름과 그 과목 test 파일에서의 행 번호(0부터). 예: `Health:44` |
| `rep` | 같은 문항을 물은 반복 번호(0~4) |
| `first` | 첫 답의 글자. 비어 있으면 답에서 글자를 뽑지 못한 것이다 |
| `answer` | 정답 글자 |
| 나머지 열 | 열 이름이 조건이고 값이 그 조건에서 받은 둘째 답의 글자다. 비어 있으면 돌리지 않은 조건이거나 글자를 뽑지 못한 답이다 |

조건 이름은 `rebut_haeyo_p1`처럼 종류, 화계, 문구 번호를 잇는다. `rebut`는 붙인 말이 없는 반박, `control`은 답만
다시 묻는 통제다. 조건마다 실제로 보낸 문장은 `src/ksyco/conditions.py`, `main_design.py`, `followup.py`에 있다.

## 분석 다시 돌리기

API 키도 비용도 필요 없다. `shared/trials.csv`만 읽는다.

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,report]"
.venv\Scripts\python tools\figures.py            # 그림 네 장과 미리 정한 비교의 구간
.venv\Scripts\python tools\followup_explore.py   # 붙인 말 묶음끼리의 차이
.venv\Scripts\python tools\factorial_explore.py  # 요인설계로 본 효과와 분산분석표
.venv\Scripts\python tools\accuracy_explore.py   # 반박 전과 뒤의 정답률
```

`report`는 분산분석과 그림에 쓰는 scipy와 matplotlib이다. 질문의 말투와 표기를 바꾼 조건, 파일럿의 응답은 이
표에 없다. 그쪽 결과는 `shared/reports/`의 집계표로만 볼 수 있다.

## 실험 다시 돌리기

Batch API로 요청을 보내므로 비용이 든다. 설계는 `docs/specs/`에 있다.

```
.venv\Scripts\python -m ksyco.pilot plan      # 요청 수만 계산한다. API를 호출하지 않는다
.venv\Scripts\python -m ksyco.pilot sample    # KMMLU 파일을 받고 문항을 뽑는다
.venv\Scripts\python -m ksyco.pilot check     # 모델이 화계를 구분하는지 확인한다
.venv\Scripts\python -m ksyco.pilot first     # 첫 답
.venv\Scripts\python -m ksyco.pilot second    # 둘째 답
.venv\Scripts\python -m ksyco.pilot report    # 저장된 응답으로 표를 만든다
```

본 실험은 `ksyco.main_run`(`plan`, `sample`, `first`, `second`, `report`), 후속 실험은 `ksyco.followup`(`plan`,
`second`, `report`)으로 같은 방식이다.

- `check`, `first`, `second`는 요청을 보내고 끝날 때까지 기다린다. 중간에 끊기면 같은 명령을 다시 실행한다. 이미 보낸 묶음은 다시 보내지 않는다.
- `--no-wait`를 붙이면 묶음을 보내기만 하고 끝낸다. 나중에 같은 명령을 `--no-wait` 없이 실행하면 결과를 받는다. 결과는 29일 동안 보관된다.
- `ksyco.pilot`에 `--smoke`를 붙이면 문항 2개로 작게 돌린다.
- `tools\share.py`는 `results/`의 응답에서 `shared/`를 다시 만든다.

### 비밀 값

| 이름 | 쓰임 | 넣는 곳 |
| --- | --- | --- |
| `KSYCO_ANTHROPIC_API_KEY` | Claude API 호출 | Windows 사용자 환경 변수 |

이 프로젝트 전용 키를 따로 만들어 쓴다. 흔히 쓰는 이름인 `ANTHROPIC_API_KEY`는 읽지 않는다. 다른 프로그램이 그
이름을 자동으로 읽어 쓰기 때문에, 어느 계정에 과금되는지 분명히 하려고 이름을 나눴다.

```
[Environment]::SetEnvironmentVariable('KSYCO_ANTHROPIC_API_KEY', '<새 키>', 'User')
```

설정한 뒤에 연 터미널부터 적용된다. 값은 파일, 출력, 기록 어디에도 적지 않는다.

## 한계

- 한 회사의 모델 둘, 과목 여섯, 4지선다 문항에서 나온 결과다.
- 글자 하나로만 답하게 했다. 설명할 기회가 있는 대화에서는 다를 수 있다.
- 반박 문구는 세 가지뿐이고 문구에 따라 번복률이 12%p쯤 달랐다.
- 후속 실험과 요인설계 분석은 같은 문항으로 결과를 본 뒤에 한 탐색이다. 새 문항으로 확인하지 않았다.

## 라이선스와 출처

- 코드는 MIT 라이선스다(`LICENSE`).
- 문항은 KMMLU에서 가져왔다. Son et al., "KMMLU: Measuring Massive Multitask Language Understanding in Korean", [arXiv:2402.11548](https://arxiv.org/abs/2402.11548). 라이선스는 CC BY-ND 4.0이다. `shared/trials.csv`의 `uid`와 `answer` 열이 이 자료에서 온 것이다.
- 참고한 연구는 `docs/literature/RELATED_WORK.md`에 정리했다.
