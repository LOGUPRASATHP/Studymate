import re
from collections import Counter
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.credentials import Credentials
import os

# Cache model instance
_cached_model = None
_pattern_cache = {}  # Cache compiled regex patterns
_watsonx_client = None

def get_watsonx_client():
    """Initialize and return Watsonx client"""
    global _watsonx_client
    if _watsonx_client is None:
        try:
            # Get credentials from environment variables
            api_key = os.getenv("WATSONX_API_KEY", "your-api-key-here")
            url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
            project_id = os.getenv("WATSONX_PROJECT_ID", "your-project-id-here")
            
            # Create credentials
            credentials = Credentials(
                api_key=api_key,
                url=url
            )
            
            # Initialize client
            _watsonx_client = APIClient(credentials)
            _watsonx_client.project_id = project_id
            
        except Exception as e:
            print(f"Error initializing Watsonx client: {e}")
            # Don't raise here - we'll fallback to local generation
            _watsonx_client = None
    
    return _watsonx_client

def get_cached_pattern(pattern):
    """Cache compiled regex patterns for faster execution"""
    if pattern not in _pattern_cache:
        _pattern_cache[pattern] = re.compile(pattern, re.IGNORECASE)
    return _pattern_cache[pattern]

def generate_answer(context_chunks, question):
    """
    Generates comprehensive, detailed answers based on the context chunks
    """
    # ✅ Limit context to 3 chunks (optimized selection)
    context = "\n".join(context_chunks[:3])

    # Fast length check - return early if insufficient context
    if not context or len(context.strip()) < 50:
        return "I couldn't extract enough text from the PDF to answer your question. Please try with a different PDF or ensure the document contains readable text."

    try:
        # First try Watsonx
        watsonx_answer = generate_watsonx_answer(context, question)
        if watsonx_answer:
            return watsonx_answer
        
        # Fallback to local generation if Watsonx fails
        return generate_detailed_answer(context, question)
    except Exception as e:
        # Final fallback
        return f"Error processing your question: {str(e)}"

def generate_watsonx_answer(context, question):
    """Generate answer using IBM Watsonx Mixtral-8x7B-Instruct model"""
    client = get_watsonx_client()
    if client is None:
        return None  # Fallback to local generation
    
    # Prepare prompt for Mixtral
    prompt = f"""<|system|>
You are an academic assistant helping students understand their study materials.
Based strictly on the provided context, answer the question comprehensively.
Provide detailed explanations, definitions, and examples when relevant.
If the context doesn't contain enough information, say so clearly.

Context from study materials:
{context}
</s>
<|user|>
{question}
</s>
<|assistant|>"""
    
    try:
        # Generate answer using Watsonx
        response = client.text_generation.create(
            model="ibm-mistralai/mixtral-8x7b-instruct-v01",
            parameters={
                "decoding_method": "sample",
                "max_new_tokens": 1024,
                "min_new_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0
            },
            inputs=[prompt]
        )
        
        # Extract the generated text
        generated_text = response.results[0].generated_text
        
        # Format the answer nicely
        return format_watsonx_answer(generated_text, question)
        
    except Exception as e:
        print(f"Watsonx API error: {str(e)}")
        return None  # Fallback to local generation

def format_watsonx_answer(answer_text, question):
    """Format the Watsonx response into a structured answer"""
    return f"""# 📚 Comprehensive Analysis: {question}

## 🔍 Answer from Your Study Material:

{answer_text}

## 💡 Study Tips:

• **Review key concepts** mentioned in the answer
• **Create flashcards** for important terms
• **Practice explaining** these concepts in your own words
• **Connect this information** to what you already know

*Generated using IBM Watsonx with Mixtral-8x7B-Instruct model*"""

def generate_detailed_answer(context, question):
    """Generate a comprehensive, detailed answer based on the actual context"""
    # Pre-compute all extractions in optimal order
    key_terms = extract_key_terms(context)
    relevant_sentences = extract_relevant_sentences(context, question)
    
    # Early return if no relevant content
    if not relevant_sentences:
        return generate_comprehensive_fallback(context, question, key_terms)
    
    # Only compute these if needed (lazy evaluation)
    definitions = extract_definitions(context) if len(relevant_sentences) < 6 else []
    examples = extract_examples(context) if len(relevant_sentences) < 8 else []
    
    # Build answer efficiently
    answer_parts = [
        f"""# 📚 Comprehensive Analysis: {question}

## 🔍 Key Findings from Your Study Material:

"""
    ]
    
    # Add relevant sentences
    for i, sentence in enumerate(relevant_sentences[:6], 1):
        answer_parts.append(f"**{i}. {sentence}**\n\n")
    
    # Add definitions if available and needed
    if definitions:
        answer_parts.append("\n## 📖 Important Definitions:\n\n")
        for i, definition in enumerate(definitions[:3], 1):
            answer_parts.append(f"**Definition {i}:** {definition}\n\n")
    
    # Add examples if available and needed
    if examples:
        answer_parts.append("\n## 💡 Practical Examples:\n\n")
        for i, example in enumerate(examples[:2], 1):
            answer_parts.append(f"**Example {i}:** {example}\n\n")
    
    # Add key concepts
    if key_terms:
        answer_parts.append(f"\n## 🎯 Key Related Concepts:\n\n")
        answer_parts.append(", ".join([f"**{term}**" for term in key_terms[:5]]))
        answer_parts.append("\n\n")
    
    # Add recommendations
    answer_parts.append(f"""\n## 📝 Study Recommendations:

1. **Focus on understanding** the relationships between {', '.join(key_terms[:2]) if key_terms else 'these concepts'}
2. **Review the definitions** and how they apply to different contexts
3. **Practice with examples** to reinforce your understanding
4. **Connect these concepts** to broader topics in your field

## 💭 Further Exploration:

For deeper understanding, consider researching how these concepts relate to real-world applications.""")
    
    # Join all parts efficiently
    answer = "".join(answer_parts)
    
    # Ensure minimum length
    if len(answer.split()) < 150:
        answer += generate_additional_insights(context, question)
    
    return answer

def generate_comprehensive_fallback(context, question, key_terms):
    """Generate a detailed fallback response when no direct matches"""
    # Pre-compute formatted topics
    formatted_topics = format_key_topics(context)
    
    return f"""# 📚 Comprehensive Analysis: {question}

## 🔍 Overview of Your Study Material:

The content primarily focuses on **{', '.join(key_terms[:3]) if key_terms else 'key concepts'}**. While I couldn't find direct information about "{question}", here's what the material covers:

## 📖 Main Topics Discussed:

{formatted_topics}

## 🎯 Suggested Study Approach:

1. **Review foundational concepts** related to the main topics
2. **Look for connections** between different concepts
3. **Examine practical applications** mentioned
4. **Note definitions and terminology** used

## 💡 Potential Related Questions:

- How does {question} relate to the main concepts?
- What applications might {question} have in this context?

## 📝 Recommendation:

Review sections discussing related concepts for indirect references to "{question}"."""

def format_key_topics(context):
    """Format key topics in a structured way"""
    sentences = re.split(r'[.!?]+', context)
    substantial_sentences = [s.strip() for s in sentences if len(s.strip()) > 50][:4]
    
    return "\n".join([f"{i}. **{sentence}**" for i, sentence in enumerate(substantial_sentences, 1)])

def extract_relevant_sentences(context, question):
    """Extract sentences relevant to the question with better matching"""
    # Pre-compile patterns
    sentence_splitter = get_cached_pattern(r'[.!?]+')
    word_pattern = get_cached_pattern(r'\b[a-zA-Z]{4,}\b')
    
    sentences = sentence_splitter.split(context)
    relevant_sentences = []
    
    # Convert question to search terms (optimized)
    question_terms = set(word_pattern.findall(question.lower()))
    stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'does', 'could', 'would', 'should'}
    question_terms = [term for term in question_terms if term not in stop_words]
    
    # Fast matching with early termination
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 30:  # Only substantial sentences
            sentence_lower = sentence.lower()
            # Check for any term match
            if any(term in sentence_lower for term in question_terms):
                relevant_sentences.append(sentence)
                if len(relevant_sentences) >= 8:  # Early termination
                    break
    
    # If no direct matches, return substantial sentences
    if not relevant_sentences:
        relevant_sentences = [s.strip() for s in sentences if len(s.strip()) > 40][:6]
    
    return relevant_sentences[:6]

def extract_definitions(context):
    """Extract definition-like sentences"""
    definition_patterns = [
        r'is defined as', r'refers to', r'means that', r'is called',
        r'known as', r'can be defined', r'is essentially', r'is described as'
    ]
    
    sentences = re.split(r'[.!?]+', context)
    definitions = []
    compiled_patterns = [get_cached_pattern(pattern) for pattern in definition_patterns]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:
            for pattern in compiled_patterns:
                if pattern.search(sentence):
                    definitions.append(sentence)
                    break
            if len(definitions) >= 3:  # Early termination
                break
    
    return definitions

def extract_examples(context):
    """Extract example sentences"""
    example_patterns = [
        r'for example', r'such as', r'for instance', r'including',
        r'like', r'e\.g\.', r'as in', r'case study'
    ]
    
    sentences = re.split(r'[.!?]+', context)
    examples = []
    compiled_patterns = [get_cached_pattern(pattern) for pattern in example_patterns]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 30:
            for pattern in compiled_patterns:
                if pattern.search(sentence):
                    examples.append(sentence)
                    break
            if len(examples) >= 2:  # Early termination
                break
    
    return examples

def extract_key_terms(text):
    """Extract key terms from text"""
    stop_words = {
        'this', 'that', 'these', 'those', 'which', 'what', 'when', 'where',
        'who', 'why', 'how', 'with', 'from', 'have', 'has', 'been', 'were',
        'are', 'and', 'the', 'for', 'not', 'but', 'their', 'will', 'would',
        'should', 'could', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'upon', 'under', 'while', 'until', 'then',
        'than', 'also', 'more', 'less', 'most', 'least'
    }
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    
    # Get most frequent terms with Counter optimization
    return [word for word, _ in Counter(filtered_words).most_common(10)]

def generate_additional_insights(context, question):
    """Generate additional insights to enrich the answer"""
    key_terms = extract_key_terms(context)[:3]
    
    return f"""

## 🔍 Additional Insights:

Based on the context, consider these points:

- The material emphasizes practical applications
- Key terminology suggests fundamental concepts
- Content builds on previous knowledge

## 📚 Recommended Next Steps:

1. Create flashcards for: {', '.join(key_terms) if key_terms else 'key terms'}
2. Summarize main sections
3. Identify conceptual connections
4. Practice explaining concepts"""