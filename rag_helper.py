from opentelemetry import trace

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()

# Price per 1M tokens (gpt-4o-mini rates)
PRICE_PER_1M_INPUT_TOKENS = 0.15
PRICE_PER_1M_OUTPUT_TOKENS = 0.60


def calculate_openai_cost(input_tokens, output_tokens):
    input_cost = input_tokens * PRICE_PER_1M_INPUT_TOKENS / 1_000_000
    output_cost = output_tokens * PRICE_PER_1M_OUTPUT_TOKENS / 1_000_000
    return input_cost + output_cost


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['filename'])
            lines.append(doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.output_text


class RAGTraced(RAGBase):
    """RAG subclass that wraps rag(), search(), and llm() methods with OpenTelemetry spans."""

    def __init__(self, *args, tracer_name="llm-zoomcamp", **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = trace.get_tracer(tracer_name)

    def search(self, query, num_results=5):
        with self.tracer.start_as_current_span("search") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_results", num_results)
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        with self.tracer.start_as_current_span("llm") as span:
            span.set_attribute("prompt_length", len(prompt))
            span.set_attribute("model", self.model)
            response = super().llm(prompt)

            usage = response.usage
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)

            cost = calculate_openai_cost(usage.input_tokens, usage.output_tokens)
            span.set_attribute("cost", cost)

            return response

    def rag(self, query):
        with self.tracer.start_as_current_span("rag") as span:
            span.set_attribute("query", query)
            return super().rag(query)
