import os
from tqdm import tqdm
import json
import re
import unicodedata
from openai import OpenAI

if __name__ == "__main__":
    pattern = r'(\{[^}]*\})'

    preamble = "안녕하세요. 저는 질문에 대한 정답의 규격을 문맥과 맞추고 질문을 수정하는 Self-Refine GPT 입니다."
    instruction = "문맥에 있는 Dictionary의 Value 중 답변과 같지만 포멧만 다른 것이 있다면 답변의 포멧을 Dictionary Value로 변경 해주시오. 질문이 구체적이지 않거나 답변 변경 후 질문이 변경된 답변과 일치하지 않는다면 질문을 구체적이게 문맥과 변경된 답변에 맞춰주시오. \"{}\"와 같은 Dictionary 형태로 출력해주시오."
    few_shot = [
        {"pos": ["급수공사 연간단가계약 일반․특별 시방서\n{\"연번\": \"6\", \"장비 및 공구명\": \"작업차량\", \"규격\": \"1.0톤이상\", \"수량\": \"1대이상\"}"], "query": "급수공사 연간단가계약 일반․특별 시방서에서 작업차량의 규격은 얼마니?",
         "question": "급수공사 연간단가계약 일반․특별 시방서에서 작업차량의 규격은 얼마니?", "answer": "1.0톤 이상", "neg": [], "chosen": "1.0톤 이상", "refined_q": "급수공사 연간단가계약 일반․특별 시방서에서 작업차량의 규격은 얼마니?", "refined_a": "1.0톤이상"},
        {"pos": ["일반용역 입찰 공고서\n공고명 적합성 인증기준(안) 제도화 지원을 위한 규제영향분석\n{\"구분\": \"입찰참가자격 확인서류 등\", \"서류명\": \"⑤ 법인등기부등본 원본\", \"유의사항\": \"제안서제출일 기준 최근 3개월 이내 발급분 개인인 경우 개인인감증명서 및 주민등록등본\"}"], "query": "입찰참가 자격 확인서류 중 법인등기부등본은 언제 발급받아야 하는가?", "question": "입찰참가 자격 확인서류 중 법인등기부등본은 언제 발급받아야 하는가?", "refined_q": "입찰참가 자격 확인서류 중 개인인 경우 법인등기부등본은 언제, 어떻게 발급받아야 하는가?", "refined_a": "제안서제출일 기준 최근 3개월 이내 발급분 개인인 경우 개인인감증명서 및 주민등록등본", "answer": "제안서제출일 기준 최근 3개월 이내 발급분이어야 한다.", "neg": [], "chosen": "제안서제출일 기준 최근 3개월 이내 발급분이어야 한다."},
        {"pos": ["- 김천대학교 덕곡동 근린생활시설 신축공사 건축감리용역 -과업이행요청서\n{\"제출건명\": \"수시보고\", \"제출 시기\": \"◦감독관지시가 있을 경우 즉시 보고\", \"첨부물\": \"◦감독관 요구 서류\", \"제출 부수\": \"2부\"}"],
         "query": "수시보고는 어떤 시기에 제출해야 해?", "question": "수시보고는 어떤 시기에 제출해야 해?", "answer": "감독관 지시가 있을 경우 즉시 보고", "neg": [], "chosen": "감독관 지시가 있을 경우 즉시 보고", "refined_a": "◦감독관지시가 있을 경우 즉시 보고",
         "refined_q": "김천대학교 덕곡동 근린생활시설 신축공사 건축감리용역 과업이행요청서에서 수시보고는 어떤 시기에 제출해야 해?"},
        {"pos": ["수성대학교 교내식당 및 편의점 위탁운영 제안요청서\n{\"연번\": \"5\", \"구분\": \"우선협상대상자 선정 및 협상실시\", \"일정\": \"2021. 11. 24.(수) ~\"}"],
         "query": "수성대학교 교내식당 및 편의점 위탁운영에 관한 제안요청서에서 우선협상대상자 선정 및 협상 실시 일정은 언제부터 시작되니?",
         "question": "수성대학교 교내식당 및 편의점 위탁운영에 관한 제안요청서에서 우선협상대상자 선정 및 협상 실시 일정은 언제부터 시작되니?", "answer": "2021. 11. 24.(수) 부터 시작된다.",
         "refined_a": "2021. 11. 24.(수) ~", "refined_q": "수성대학교 교내식당 및 편의점 위탁운영에 관한 제안요청서에서 우선협상대상자 선정 및 협상 실시 일정은 언제부터 시작되니?",
         "neg": [], "chosen": "2021. 11. 24.(수) 부터 시작된다."},
        {"pos": ["제안요청서\n사업명 『경력단절예방 지원사업 효과성 분석』\n{\"비목\": \"3) 전산처리비\", \"계상기준\": \"◦당해연구내용과 관련된 자료처리를 위한 외부컴퓨터 사용료 및 부대비용 ◦자산가치가 있는 S/W 및 H/W 구입비는 계상할 수 없음\"}"],
         "query": "전산처리비는 어떤 용도로 계상할 수 있고, 어떤 항목은 계상할 수 없나요?", "question": "전산처리비는 어떤 용도로 계상할 수 있고, 어떤 항목은 계상할 수 없나요?",
         "refined_a": "◦당해연구내용과 관련된 자료처리를 위한 외부컴퓨터 사용료 및 부대비용 ◦자산가치가 있는 S/W 및 H/W 구입비는 계상할 수 없음", "refined_q": "전산처리비는 어떤 용도로 계상할 수 있고, 어떤 항목은 계상할 수 없나요?",
         "answer": "전산처리비는 당해 연구내용과 관련된 자료처리를 위한 외부컴퓨터 사용료 및 부대비용으로 계상할 수 있으며, 자산가치가 있는 S/W 및 H/W 구입비는 계상할 수 없습니다.", "neg": [], "chosen": "전산처리비는 당해 연구내용과 관련된 자료처리를 위한 외부컴퓨터 사용료 및 부대비용으로 계상할 수 있으며, 자산가치가 있는 S/W 및 H/W 구입비는 계상할 수 없습니다."}
    ]
    for i in few_shot:
        pos = i["pos"][0]
        json_pos = json.loads(re.findall(pattern, pos)[0])
        values = set(list(json_pos.values()))
        if i['refined_a'] not in values:
            raise()
    few_shot = "\n\n\n\n".join([f"[질문]: {f['question']}\n[정보]: {f['pos'][0]}\n[답변]: {f['answer']}\n=> {{ \"수정 답변\": \"{f['refined_a']}\", \"수정 질문\": \"{f['refined_q']}\"}}" for f in few_shot])

    client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
    """
    First Refine
    f = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_wrong.jsonl", "r")
    w = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_wrong_refine.jsonl", "w", encoding="utf-8")
    e = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_wrong_refine_error.jsonl", "w", encoding="utf-8")

    """
    f = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_wrong_refine_error.jsonl", "r")
    w = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_wrong_refine_error_refine.jsonl", "w", encoding="utf-8")
    e = open("/mnt/c/Users/thddm/Documents/dataset/synthetic_qa_v2_wrong_refine_error_refine_error.jsonl", "w", encoding="utf-8")

    for i in tqdm(f):
            i = json.loads(i)

            message = [{"role": "system", "content": unicodedata.normalize("NFC", f"{preamble}\n{instruction}\n{few_shot}\n\n\n\n")}, {"role": "user", "content": unicodedata.normalize("NFC", f"위의 예시를 참고하여 {instruction}\n[질문]: {i['question']}\n[정보]: {i['pos'][0]}\n[답변]: {i['answer']}\n=> ")}]
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=message,
                    temperature = 0.5,
                    top_p =0.5
                )

                out = json.loads(re.findall(pattern, completion.choices[0].message.content)[0])
                i['question'] = out['수정 질문']
                i['query'] = i['question']
                i['answer'] = out["수정 답변"]
                i['chosen'] = out["수정 답변"]
                pos = i["pos"][0]
                json_pos = json.loads(re.findall(pattern, pos)[0])
                values = set(list(json_pos.values()))
                if i['answer'] not in values:
                    raise()
                w.write(json.dumps(i, ensure_ascii=False) + "\n")
            except:
                None
                e.write(json.dumps(i, ensure_ascii=False) + "\n")