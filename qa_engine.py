from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

def generate_answer(context_chunks, question):
    # Combine context chunks into a single string
    context = "\n".join(context_chunks)

    # Construct the prompt
    prompt = (
        f"Answer the question based on the context below:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    # IBM Watsonx credentials
    creds = Credentials(
        api_key="jyquknJalB5nmMSzTRY1MzeIM8Oz8xoGAcajignambpj",
        url="https://au-syd.ml.cloud.ibm.com"
    )

    # Generation parameters
    gen_params = {
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY
    }

    # Model setup
    model = ModelInference(
        model_id="mistralai/mistral-large",
        credentials=creds,
        project_id="84709837-2d6a-4001-aabf-85152bafe09b",
        params=gen_params
    )

    # Generate the response
    response = model.generate(prompt=prompt)

    # Safely extract the generated text
    if response.get('results') and 'generated_text' in response['results'][0]:
        return response['results'][0]['generated_text']
    else:
        return "No answer generated. Please check model response or parameters."
