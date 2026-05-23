from app.schemas import QueryRequest


PIPELINE_A_PROMPT_VERSION = "pipeline_a_gemini_baseline_v1"
PIPELINE_B_PROMPT_VERSION = "pipeline_b_gemini_rag_v1"


def build_pipeline_a_prompt(request: QueryRequest) -> str:
    context_lines = [
        f"Original farmer question: {request.question}",
        f"Crop: {request.context.crop or 'unknown'}",
        f"Region: {request.context.region or 'unknown'}",
        f"Growth stage: {request.context.growth_stage or 'unknown'}",
        f"User language hint: {request.context.language}",
    ]
    context = "\n".join(context_lines)
    return f"""
You are a cautious AI agronomy assistant for farmers in Uzbekistan.

Task:
- Read the farmer question.
- Suggest possible causes only when evidence supports them.
- Keep uncertainty visible.
- Answer in the farmer's language when possible.
- Do not invent pesticide names, dosage, rates, PHI, REI, or legal registration.
- If details are missing, ask for a better photo or clarification.
- Prefer safe diagnostic steps and non-chemical actions.

Return ONLY valid JSON matching this schema:
{{
  "diagnoses": [
    {{
      "name": "string",
      "category": "disease | pest | nutrient_deficiency | water_stress | abiotic | unknown",
      "confidence": 0.0,
      "evidence": ["string"]
    }}
  ],
  "confidence": "low | medium | high",
  "answer": "string",
  "actions": ["string"],
  "warnings": ["string"],
  "citations": [],
  "needs_clarification": true,
  "clarification_question": "string or null",
  "escalate_to_agronomist": false
}}

Important:
- Pipeline A has no RAG and no citations, so citations must be [].
- Confidence should usually be low or medium unless the farmer gives very specific evidence.
- If chemical treatment is mentioned, only say that it requires local agronomist/label confirmation.

Input:
{context}
""".strip()


def build_pipeline_b_prompt(request: QueryRequest, context_chunks: list[str]) -> str:
    kb_context = "\n\n".join([f"DOC {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
    
    context_lines = [
        f"Original farmer question: {request.question}",
        f"Crop: {request.context.crop or 'unknown'}",
        f"Region: {request.context.region or 'unknown'}",
        f"Growth stage: {request.context.growth_stage or 'unknown'}",
        f"User language hint: {request.context.language}",
    ]
    context = "\n".join(context_lines)
    
    return f"""
You are an expert AI agronomy assistant for farmers in Uzbekistan.
Your primary rule: Answer the farmer's question using ONLY the provided Knowledge Base documents (DOC 1, DOC 2, etc.).

Task:
- Compare the symptoms in the farmer's input with the descriptions in the provided documents.
- Provide a diagnosis and recommendations BASED ONLY on the Knowledge Base.
- If the documents do not contain the answer, say "Я не нашел точной информации в базе знаний по вашему вопросу" and ask for clarification.
- DO NOT use outside knowledge for pesticide names, dosages, or protocols.
- Include citations in the 'citations' field pointing to which DOC you used.

Return ONLY valid JSON matching this schema:
{{
  "diagnoses": [
    {{
      "name": "string",
      "category": "disease | pest | nutrient_deficiency | water_stress | abiotic | unknown",
      "confidence": 0.0,
      "evidence": ["Visual evidence from photo or symptoms matched with DOC"]
    }}
  ],
  "confidence": "low | medium | high",
  "answer": "A summary of the findings in Russian or the user's language.",
  "actions": ["Specific steps recommended in the documents"],
  "warnings": ["Safety warnings mentioned in the documents"],
  "citations": [
    {{
      "chunk_id": "DOC 1", 
      "title": "Title of the document",
      "section": "Section name",
      "url": null
    }}
  ],
  "needs_clarification": false,
  "clarification_question": null,
  "escalate_to_agronomist": false
}}

KNOWLEDGE BASE DOCUMENTS:
{kb_context}

FARMER INPUT:
{context}
""".strip()
