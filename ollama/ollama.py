import requests
import json
import time
import statistics


def compute_itl_stats(token_timestamps_ns: list[int], eval_count: int = 0) -> dict:
    """
    Calcula métricas de latencia entre chunks de red (ITL) y el TPOT real client-side.

    ITL_i = tiempo entre la llegada del chunk HTTP i y el chunk i+1.
    TPOT client-side = Tiempo total de decodificación / (Tokens reales - 1).

    Tiene en cuenta que pueden venir varios tokens en 1 chunk por como va Ollama.
    """
    empty = {
        "itl_mean": None,
        "itl_median": None,
        "itl_p95": None,
        "itl_std": None,
        "tpot_client": None,
    }
    
    if len(token_timestamps_ns) < 2:
        return empty

    itls = [
        (token_timestamps_ns[i] - token_timestamps_ns[i - 1]) / 1e9
        for i in range(1, len(token_timestamps_ns))
    ]
    sorted_itls = sorted(itls)
    p95_idx = max(0, min(int(len(sorted_itls) * 0.95) - 1, len(sorted_itls) - 1))
    tokens_generated = eval_count if eval_count > 1 else len(token_timestamps_ns)
    total_decode_time_client = (token_timestamps_ns[-1] - token_timestamps_ns[0]) / 1e9
    true_tpot_client = total_decode_time_client / (tokens_generated - 1)

    return {
        "itl_mean": statistics.mean(itls),
        "itl_median": statistics.median(itls),
        "itl_p95": sorted_itls[p95_idx],
        "itl_std": statistics.stdev(itls) if len(itls) > 1 else 0.0,
        "tpot_client": true_tpot_client, # TPOT desvinculado de los chunks de red
    }


class Ollama():
    """
    Clase base para la comunicación con la API REST de Ollama.
    
    Proporciona una interfaz simplificada para interactuar con el servidor Ollama,
    encapsulando las llamadas HTTP a los diferentes endpoints de la API para
    gestión de modelos, generación de texto y operaciones de chat.
    
    Attributes:
        url (str): URL base del servidor Ollama construida con IP y puerto
    """
    def __init__(self, host_ip: str, host_port: str = "11434"):
        """
        Inicializa la conexión con el servidor Ollama.
        
        Args:
            host_ip (str): Dirección IP del servidor Ollama
            host_port (str, optional): Puerto del servidor Ollama. Por defecto "11434"
        """
        self.url = f"http://{host_ip}:{host_port}"
    
    def chat_prompt(self, model: str, prompts: list) -> requests.Response:
        """
        Realiza una conversación multi-mensaje con un modelo LLM.
        
        Utiliza el endpoint /api/chat para mantener contexto conversacional
        entre múltiples intercambios de mensajes.
        
        Args:
            model (str): Nombre del modelo a utilizar
            prompts (list): Lista de mensajes de la conversación
            
        Returns:
            requests.Response: Respuesta HTTP del servidor Ollama
        """
        response = requests.post(
            f"{self.url}/api/chat",
            json = {
                "model" : model,
                "messages" : prompts,
                "stream" : False
            }
        )

        return response

    def load_model(self, model: str):
        """
        Carga un modelo en memoria para optimizar respuestas posteriores.
        
        Args:
            model (str): Nombre del modelo a cargar
            
        Returns:
            requests.Response: Respuesta HTTP del servidor
        """
        return requests.post(f"{self.url}/api/generate")
    
    def single_prompt(self, model: str, prompt: str) -> requests.Response:
        """
        Ejecuta un prompt simple contra un modelo LLM.
        
        Utiliza el endpoint /api/generate para generar texto basado en un
        prompt único sin contexto conversacional previo.
        
        Args:
            model (str): Nombre del modelo a utilizar
            prompt (str): Texto del prompt a procesar
            
        Returns:
            requests.Response: Respuesta HTTP con la generación del modelo
        """
        response = requests.post(
            f"{self.url}/api/generate",
            json = {
                "model" : model,
                "prompt" : prompt,
                "stream" : False #inside JSON is for receiving the response by chunks/token by token, if false the total response is send in one time
            }
        )

        return response
    
    def single_prompt_with_ttft(self, model: str, prompt: str, tg_tokens: int = -1):
        """
        Ejecuta un prompt midiendo TTFT usando streaming.

        Ollama solo entrega el primer token de forma observable cuando stream=True.
        Se reconstruye la respuesta completa y se conserva el JSON final de Ollama
        para mantener las métricas agregadas existentes, habría que repasar esto.
        """
        start_time = time.perf_counter_ns()

        #Para que si no se introduce tg se use el anterior json, si se introduce se usa el nuevo
        if(tg_tokens < 1):
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True
            )
        else:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": tg_tokens  #Tokens generados
                    }
                },
                stream=True
            )

        if response.status_code != 200:
            return response, None

        first_token_time = None
        token_timestamps_ns = []
        response_chunks = []
        final_data = {}

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            data = json.loads(line)
            chunk = data.get("response", "")

            if chunk:
                now = time.perf_counter_ns()
                if first_token_time is None:
                    first_token_time = now
                token_timestamps_ns.append(now)

            response_chunks.append(chunk)

            if data.get("done"):
                final_data = data

        final_data["response"] = "".join(response_chunks)
        final_data["ttft_duration"] = (
            first_token_time - start_time if first_token_time is not None else None
        )
        
        eval_count = final_data.get("eval_count", 0)
        final_data.update(compute_itl_stats(token_timestamps_ns, eval_count))

        return response, final_data

    
    def pull_model(self, model: str) -> None:
        """
        Descarga un modelo desde el repositorio de Ollama.
        
        Utiliza el endpoint /api/pull para descargar modelos desde
        la librería oficial de Ollama. Incluye validación de respuesta.
        
        Args:
            model (str): Nombre del modelo a descargar
            
        Raises:
            Exception: Si la descarga falla o el modelo no existe
        """
        response = requests.post(
            f"{self.url}/api/pull",
            json = {"name": model},
            stream = False # Stream out the json is for wait to the total response instead of recieving by chunks/token by token
        )
        if response.status_code != 200:
            raise Exception(f"Impossible to download model: {model}, you can check if the model exists in : https://ollama.com/library")
    
    def get_models(self) -> requests.Response:
        """
        Obtiene la lista de modelos disponibles en el servidor.
        
        Returns:
            requests.Response: Respuesta con la lista de modelos instalados
        """
        response  = requests.get(
            f"{self.url}/api/tags"
        )
        
        return response
    
    def list_models(self) -> None: # Function for debugging purposes
        """
        Imprime la lista de modelos disponibles en el servidor.
        
        Función de utilidad para debugging que muestra por consola
        todos los modelos instalados en el servidor Ollama.
        """
        response = self.get_models()
        data = response.json()
        models = data.get("models")
        for model in models:
            print(model.get("name"))