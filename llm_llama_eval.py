import sys

import click 
import re
import requests
import os
import paramiko
import time
import datetime
import shutil
import platform
import paramiko.ssh_exception
from logger.log import logger
from ollama.ollama_handler import OllamaHandler
from prometheus.prometheus_handler import PrometheusHandler
WORKING_PATH = "llm-eval"
OLLAMA_PATH = f"{WORKING_PATH}/ollama"
EXECUTION_PATH = os.path.dirname(os.path.realpath(__file__))
START_TIME = datetime.datetime.fromtimestamp(time.time())
    

def run_command(ssh: paramiko.SSHClient, command: str) -> tuple[paramiko.ChannelFile, paramiko.ChannelFile, paramiko.ChannelFile]:
    """
    Ejecuta un comando remoto via SSH con logging y sincronización.
    
    Proporciona una interfaz unificada para ejecución de comandos remotos
    con timeout configurado, logging detallado y espera síncrona del resultado.
    
    Args:
        ssh (paramiko.SSHClient): Cliente SSH conectado
        command (str): Comando a ejecutar en el sistema remoto
        
    Returns:
        tuple: (stdin, stdout, stderr) del comando ejecutado
    """
    logger.debug_color(f"Running command: {command}")
    stdin,stdout,stderr = ssh.exec_command(command, timeout=180)
    #wait for the end of the command on the remote machine
    status_code = stdout.channel.recv_exit_status()
    logger.debug_color(f"\[Command: {command}] Executed with status code = {status_code}")

    return (stdin,stdout,stderr)

def connect_with_private_key(ssh: paramiko.SSHClient, user: str, ip_address: str, private_key_file: str) -> None:
    """
    Establece conexión SSH utilizando autenticación por clave privada.
    
    Método de autenticación primario que utiliza claves RSA para conexión segura.
    Implementa manejo específico de excepciones SSH para diagnóstico detallado.
    
    Args:
        ssh (paramiko.SSHClient): Cliente SSH a configurar
        user (str): Nombre de usuario para la conexión
        ip_address (str): Dirección IP del servidor destino
        private_key_file (str): Ruta al archivo de clave privada
        
    Raises:
        paramiko.AuthenticationException: Error de autenticación
        paramiko.SSHException: Error en el protocolo SSH
        Exception: Otros errores de conexión
    """
    try:
        logger.debug_color(f"Connecting with the server using file: {private_key_file}")
        pkey = paramiko.RSAKey.from_private_key_file(private_key_file)
        ssh.connect(ip_address, username = user, pkey = pkey)

    except paramiko.AuthenticationException as e:
        logger.warning_color(f'Error on the authentication using private key: {e}')
        raise
    
    except paramiko.SSHException as e:
        logger.warning_color(f'Error in the ssh protocol using private key: {e}')
        raise
    
    except Exception as e:
        logger.warning_color(f'Error when trying to connect using private key: {e}')
        raise


def connect_with_password(ssh: paramiko.SSHClient, user: str, password: str, ip_address: str):
    """
    Establece conexión SSH utilizando autenticación por contraseña.
    
    Método de autenticación fallback cuando la clave privada no está disponible
    o falla la autenticación por clave. Proporciona robustez en la conectividad.
    
    Args:
        ssh (paramiko.SSHClient): Cliente SSH a configurar
        user (str): Nombre de usuario para la conexión
        password (str): Contraseña del usuario
        ip_address (str): Dirección IP del servidor destino
        
    Raises:
        paramiko.AuthenticationException: Error de autenticación
        paramiko.SSHException: Error en el protocolo SSH
        Exception: Otros errores de conexión
    """
    try:
        logger.debug_color('Connecting with the server using password ****')
        ssh.connect(ip_address, username = user, password = password)

    except paramiko.AuthenticationException as e:
        logger.warning_color(f'Error on the authentication using password: {e}')
        raise

    except paramiko.SSHException as e:
        logger.warning_color(f'Error in the ssh protocol using password: {e}')
        raise
    
    except Exception as e:
        logger.warning_color(f'Error when trying to connect using password: {e}')
        raise 
    
def connection_establishment(user: str, password: str, ip_address: str, private_key_file: str) -> paramiko.SSHClient:
    """
    Establece conexión SSH robusta con autenticación dual.
    
    Implementa patrón de autenticación con fallback: intenta primero clave privada
    y en caso de fallo recurre a autenticación por contraseña. Configura políticas
    de seguridad SSH y manejo de claves de host.
    
    Args:
        user (str): Nombre de usuario para la conexión
        password (str): Contraseña (puede estar vacía si solo se usa clave privada)
        ip_address (str): Dirección IP del servidor destino
        private_key_file (str): Ruta al archivo de clave privada
        
    Returns:
        paramiko.SSHClient: Cliente SSH conectado y listo para uso
        
    Raises:
        Exception: Si ambos métodos de autenticación fallan
    """
    logger.debug_color("Starting connection....")

    ssh = paramiko.SSHClient()
    #If is the first time connecting to a new server, will automatically save the public key into the .ssh/known_hosts file
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()

    try:
        connect_with_private_key(ssh, user = user, ip_address = ip_address, private_key_file = private_key_file)

    except (paramiko.AuthenticationException, paramiko.SSHException, Exception):
        logger.warning_color('Retrying...')

        try:
            if not password:
                raise Exception(f"Impossible to connect with SSH to {ip_address}")
            
            connect_with_password(ssh = ssh, user = user, password = password, ip_address = ip_address)
            

        except (paramiko.AuthenticationException, paramiko.SSHException, Exception):
            raise Exception(f"Impossible to connect with SSH to {ip_address}")
    
    logger.debug_color("Connection established!")

    return ssh

def is_gpu_available(ssh):
    """
    Detecta disponibilidad de GPU NVIDIA en el sistema remoto.
    
    Utiliza script Python especializado para detección precisa de hardware GPU.
    La detección se basa en el código de salida del script detector.
    
    Args:
        ssh (paramiko.SSHClient): Cliente SSH conectado al sistema remoto
        
    Returns:
        bool: True si hay GPU NVIDIA disponible, False en caso contrario
    """

    logger.info_color("Checking if GPU is available")

    _, stdout, _ = run_command(ssh, f"{WORKING_PATH}/venv/bin/python3 {WORKING_PATH}/gpu_exporter/detect_gpu.py")

    if stdout.channel.recv_exit_status() == 0:
        logger.info_color("GPU is available")
        return True
    
    logger.warning_color("No GPU detected")
    return False

def copy_file_in_sut(sftp: paramiko.SFTPClient, local_path: str, remote_path: str):
    """
    Transfiere un archivo del sistema local al sistema remoto via SFTP.
    
    Proporciona transferencia segura de archivos con manejo de errores
    específicos para operaciones SFTP y SSH.
    
    Args:
        sftp (paramiko.SFTPClient): Cliente SFTP activo
        local_path (str): Ruta del archivo local a transferir
        remote_path (str): Ruta destino en el sistema remoto
        
    Raises:
        Exception: Si la transferencia falla por errores de SFTP o SSH
    """
    try:
        sftp.put(local_path, remote_path)
    
    except (paramiko.sftp.SFTPError,paramiko.SSHException) as e:
        print("Error general de SSH/SFTP:", e)
        raise Exception(f'General error in SSH/SFTP: {e}')
    
    except (Exception) as e:
        raise Exception(f'Unexpected error {e}')
 
def environment_configuration(ssh: paramiko.SSHClient, password: str, ollama_version: str, node_version: str, reinstall_ollama: bool) -> None:
    """
    Configura el entorno completo en el sistema remoto para evaluación de LLMs.
    
    Transfiere archivos necesarios, crea entornos virtuales Python, instala
    dependencias y ejecuta scripts de configuración para Ollama y exporters.
    Proceso crítico que prepara toda la infraestructura de evaluación.
    
    Args:
        ssh (paramiko.SSHClient): Cliente SSH conectado
        password (str): Contraseña para operaciones que requieren sudo
        ollama_version (str): Versión específica de Ollama a instalar
        node_version (str): Versión específica de node_exporter a instalar
        reinstall_ollama (bool): Forzar reinstalación de Ollama si ya existe
        
    Raises:
        Exception: Si falla la transferencia de archivos o configuración
    """
    logger.debug_color("\[+] Setting the environment[/]")

    #primero tenemos que definir la ruta donde vamos a trabajar en el server remoto en este caso va a ser ${HOME}/llm-eval/
    run_command(ssh, f"mkdir -p {WORKING_PATH}/gpu_exporter")

    #list of files to copy and where
    files_to_copy = (
        ("configurations.sh", "configurations.sh"),
        ("gpu_exporter/requirements.txt", "gpu_exporter/requirements.txt"),
        ("gpu_exporter/detect_gpu.py", "gpu_exporter/detect_gpu.py"),
        ("gpu_exporter/gpu_export_metrics.py", "gpu_exporter/gpu_export_metrics.py")
    )
    try:
        with ssh.open_sftp() as sftp:
            for local_file, remote_file in files_to_copy:
                local_path = os.path.join(EXECUTION_PATH, local_file)
                remote_path = os.path.join(WORKING_PATH, remote_file)

                copy_file_in_sut(sftp= sftp, local_path = local_path, remote_path = remote_path)

    except Exception:
        raise
    #Ejecutamos el script configurations.sh en el servidor
    run_command(ssh, f"chmod 777 {WORKING_PATH}/gpu_exporter/*")
    run_command(ssh,f"python3 -m venv {WORKING_PATH}/venv")
    run_command(ssh,f"{WORKING_PATH}/venv/bin/pip3 install -r {WORKING_PATH}/gpu_exporter/requirements.txt")
    run_command(ssh, f"chmod 755 {WORKING_PATH}/configurations.sh")
    run_command(ssh, f'{WORKING_PATH}/configurations.sh "{password}" {ollama_version} {node_version} {reinstall_ollama}')

    logger.debug_color("\[+] Configured environment [/]")

def gpu_exporter_configuration(ssh: paramiko.SSHClient):
    """
    Inicia el exportador de métricas GPU en background.
    
    Lanza el script gpu_export_metrics.py como proceso daemon
    para recolección continua de métricas de GPU NVIDIA.
    
    Args:
        ssh (paramiko.SSHClient): Cliente SSH conectado al sistema remoto
    """
    #arrancamos el script de exportacion
    run_command(ssh, f"{WORKING_PATH}/venv/bin/python3 {WORKING_PATH}/gpu_exporter/gpu_export_metrics.py > pepe.txt 2>&1 &")

def extract_general_info_from_sut(ssh: paramiko.SSHClient, ip_address: str, gpu_available: bool):
    """
    Extrae información completa del hardware y sistema operativo del SUT.
    
    Recolecta metadatos críticos del sistema bajo prueba incluyendo:
    OS, CPU, memoria total y información de GPU si está disponible.
    Esta información caracteriza completamente el entorno de evaluación.
    
    Args:
        ssh (paramiko.SSHClient): Cliente SSH conectado
        ip_address (str): IP del sistema para incluir en metadatos
        gpu_available (bool): Si debe recolectar información de GPU
        
    Returns:
        dict: Diccionario con toda la información del sistema
    """
    data = {}

    #Read info from the SO
    _, stdout, _ = run_command(ssh, "cat /etc/os-release")
    output  = stdout.read().decode()
    for line in output.splitlines():
        line = line.split("=",1)

        if "URL" not in line[0]:
            data[line[0].strip()] = line[1].strip().strip('"') 


    #Read info from the CPU
    _, stdout, _ = run_command(
        ssh,
        'grep -m 1 "model name" /proc/cpuinfo | cut -d ":" -f2'
        )
    output = stdout.read().decode().strip()
    data["CPU_MODEL"] = output 

    #TOTAL MEMORY in GB with 2 decimals
    _, stdout, _ = run_command(
        ssh,
        'cat /proc/meminfo |  grep -m 1 "MemTotal" | cut -d ":" -f2 | xargs | cut -d " " -f1'
        )
    output = stdout.read().decode()
    try:
        mem_kb = int(output)
        data["TOTAL_MEMORY"] = f'{(mem_kb / 1024 / 1024):.2f}GB'
    
    except ValueError:
        data["TOTAL_MEMORY"] = "Unknown"

    #Ip from the SUT
    data["IP_ADDRESS"] = ip_address

    if gpu_available:
        _, stdout, _ = run_command(ssh, 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')
        output = stdout.read().decode()
        
        data["GPU_INFO"] = output if output else "Unknown"
    
    return data

def save_general_info(ssh, ip_address, gpu_available):
    """
    Persiste la información general del sistema en archivo de texto.
    
    Coordina la extracción y almacenamiento de metadatos del sistema
    en formato clave=valor para fácil procesamiento posterior.
    
    Args:
        ssh (paramiko.SSHClient): Cliente SSH conectado
        ip_address (str): Dirección IP del sistema bajo prueba
        gpu_available (bool): Disponibilidad de GPU en el sistema
    """

    data = extract_general_info_from_sut(ssh, ip_address, gpu_available)

    with open(f'{EXECUTION_PATH}/metrics/general_info.txt', "w") as f:
        for key, value in data.items():
            f.write(f'{key} = {value}\n')


def copy_file(src: str, dst: str):
    """
    Copia un archivo del origen al destino con manejo de errores.
    
    Utility para operaciones de copia de archivos locales con
    logging detallado de errores para debugging.
    
    Args:
        src (str): Ruta del archivo origen
        dst (str): Ruta del archivo destino
    """

    try:
        shutil.copyfile(src = src, dst = dst)

    except FileNotFoundError:
        logger.exception_color(f"ERROR file: {src} not found, impossible to copy")

    except Exception as e:
        logger.exception_color(f"Unexpected error while copying {src}, to {dst} : {e}")

def save_experiment():
    """
    Archiva todos los resultados del experimento en directorio timestampeado.
    
    Crea un snapshot completo del experimento incluyendo métricas de Ollama,
    Prometheus, información del sistema, respuestas de modelos, puntuaciones
    y logs para análisis histórico y reproducibilidad.
    
    El directorio se nombra con timestamp de inicio para evitar conflictos
    y facilitar organización temporal de experimentos.
    """
    fixed_time = str(START_TIME).replace(" ","-").replace(":","-").split(".")[0]

    logger.debug_color("Saving experiments results...")

    #Defining the paths to the targets
    experiment_dir = os.path.join(EXECUTION_PATH, "experiment_results",fixed_time)

    #Tuple of tuples with (subpath, file_name)
    files_to_copy = (
        ("metrics", "ollama_metrics.csv"),
        ("metrics", "prometheus_metrics.csv"),
        ("metrics", "general_info.txt"),
        ("metrics", "response.txt"),
        ("metrics", "models_score.csv"),
        ("logger", "logs.txt")
    )
    os.makedirs(experiment_dir, exist_ok = True)
    
    for subpath, file_name in files_to_copy:
        src =  os.path.join(EXECUTION_PATH, subpath, file_name)
        dst = os.path.join(experiment_dir, file_name)
        copy_file(src,dst)

    logger.debug_color("Experiments results saved!")
    

def clean_local_resources():
    """
    Limpia recursos y procesos locales del sistema de evaluación.
    
    Termina procesos de Prometheus y otros servicios locales que
    pudieran quedar ejecutándose. Incluye manejo multiplataforma
    para Windows y sistemas Unix.
    """
    logger.debug_color("Cleaning up local resources...")

    if platform.system() =="Windows":
        os.system('taskkill /PID 9090 /F')
    else:
        os.system('pkill -f prometheus')
    
    logger.debug_color("Local resources cleaned!")

def clean_sut_resources(ssh: paramiko.SSHClient):
    """
    Limpia procesos y recursos en el sistema bajo prueba remoto.
    
    Termina todos los procesos relacionados con la evaluación:
    Ollama, node_exporter y gpu_exporter para dejar el sistema
    en estado limpio tras la evaluación.
    
    Args:
        ssh (paramiko.SSHClient): Cliente SSH conectado al SUT
    """
    logger.debug_color("Cleaning up SUT resources...")

    process_to_kill = ["ollama", "node_exporter", "gpu_exporter"]
    for process in process_to_kill:
        run_command(ssh, f'pkill -f {process}')
    
    
    logger.debug_color("SUT resources cleaned!")

#Funcion llamada por el callback para validar/procesar la dir ip
def validarIp(ctx,param,valor: str) -> str:
    """
    Valida formato de dirección IP usando expresiones regulares.
    
    Callback de Click que verifica que la IP esté en formato válido
    o sea 'localhost'. Utilizado para validación en tiempo real de CLI.
    
    Args:
        ctx: Contexto de Click (no utilizado)
        param: Parámetro de Click (no utilizado)  
        valor (str): Valor de IP a validar
        
    Returns:
        str: IP validada en minúsculas
        
    Raises:
        click.BadParameter: Si el formato de IP es inválido
    """
    valor = valor.lower() # Parseamos el tipo de valo
    pattern = "(^((25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.){3}((25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?))$|localhost)"

    if not re.match(pattern,valor):
        raise click.BadParameter("El formato de la ip debe de ser el siguiente [0-255].[0-255].[0-255].[0-255] OR localhost")
    
    return valor



def check_ollama_version_exists(version: str) -> bool:
    """
    Comprueba contra la API de GitHub si la versión existe realmente.
    Ya que de momento solo se comprueba que tiene formato de versión,
    hay que comprobar que esta existe en el repositorio de Ollama.
    """
    url = f"https://api.github.com/repos/ollama/ollama/releases/tags/{version}"
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        logger.warning_color(f"No se encontro la versión de Ollama \"{version}\" para descargar")
        return True

def validate_ollama_version(ctx,param,valor: str) -> str:
    """
    Valida que la versión de Ollama especificada tenga formato válido.

    Como no es sostenible mantener una lista de versiones compatibles de Ollama
    se va a comprobar el formato de versión con expresiones regulares. 
    El formato debe ser "vx.x.x" donde x es un número entero.
    
    Args:
        ctx: Contexto de Click (no utilizado)
        param: Parámetro de Click (no utilizado)
        valor (str): Versión de Ollama a validar
        
    Returns:
        str: Versión validada
        
    Raises:
        click.BadParameter: Si la versión tiene formato inválido
    """
    if not re.match(r"^v\d+\.\d+\.\d+$", valor):
        raise click.BadParameter("La versión de Ollama debe tener formato vX.Y.Z, por ejemplo v0.11.4")
    
    return valor

def validate_node_exporter_version(ctx,param,valor: str) -> str:
    """
    Valida que la versión de node_exporter especificada esté soportada.
    
    Verifica contra lista de versiones válidas para garantizar
    compatibilidad con el sistema de métricas de Prometheus.
    
    Args:
        ctx: Contexto de Click (no utilizado)
        param: Parámetro de Click (no utilizado)
        valor (str): Versión de node_exporter a validar
        
    Returns:
        str: Versión validada
        
    Raises:
        click.BadParameter: Si la versión no está soportada
    """
    node_exporter_versions = ''
    with open(f"{EXECUTION_PATH}/versions/node_exporter_versions.txt","r") as versions:
        node_exporter_versions = [version.rstrip("\n") for version in versions]
    if valor not in node_exporter_versions:
        raise click.BadParameter(f"Error, version should be one of the followings: {node_exporter_versions}")
    return valor

@click.command()
@click.option("--user", "-u", help="Name of the user to connect in target destination",default = "root")
@click.option("--password", "-p",help="Password of the user to connect in target destination", default = "")
@click.option("--ip-address", "-i", required=True, callback=validarIp, help="Ip-Address of the host where the test it's going to be executed")
@click.option("--ollama-version", "-ov", callback=validate_ollama_version, help="ollama version to install in the SUT you must put the \"vx.x.x\"", default = "v0.24.0")
@click.option("--node-version", "-nv", callback=validate_node_exporter_version, help="node_exporter version to install in the SUT you must put the \"vx.x.x\"", default = "v1.9.1")
@click.option("--private-key", "-pk", help="Path to private key(including the name) in .pem format for ssh authentication", default=f"{os.getenv('HOME')}/.ssh/id_rsa")
@click.option("--reinstall-ollama", "-ro", is_flag=True, help="Force reinstallation of Ollama even if it's already installed", default=False)
def procesarLLM(ip_address: str, private_key: str, user: str, password: str, ollama_version: str, node_version: str, reinstall_ollama: bool):
    """
    Función principal que orquesta todo el proceso de evaluación de LLMs.
    
    Coordina el flujo completo: conexión SSH, configuración del entorno,
    detección de hardware, inicialización de servicios, sincronización
    de la evaluación con recolección de métricas, y archivado de resultados.
    
    Implementa manejo robusto de errores con cleanup garantizado de recursos
    tanto locales como remotos mediante bloques try/except/finally.
    
    Args:
        ip_address (str): IP del sistema bajo prueba
        private_key (str): Ruta a clave privada SSH
        user (str): Usuario para conexión SSH
        password (str): Contraseña (opcional si se usa clave privada)
        ollama_version (str): Versión específica de Ollama a instalar
        node_version (str): Versión específica de node_exporter a instalar
        reinstall_ollama (bool): Forzar reinstalación de Ollama
    """
    ssh = None
    ollama = None
    prometheus = None

    logger.info_color(f"Verificando disponibilidad de la versión de Ollama: {ollama_version}")
    if not check_ollama_version_exists(ollama_version):
        logger.exception_color(f"Error: La versión {ollama_version} no existe en los repositorios oficiales de Ollama.")
        sys.exit(1) # Quizás hay otra forma mejor de salir del sistema.

    try:
        ssh = connection_establishment(user, password, ip_address, private_key)
        environment_configuration(ssh, password, ollama_version, node_version, reinstall_ollama)
        gpu_available = is_gpu_available(ssh)
        if gpu_available:
            gpu_exporter_configuration(ssh)
        save_general_info(ssh, ip_address, gpu_available)
        run_command(ssh, f"OLLAMA_HOST=0.0.0.0 {OLLAMA_PATH}/bin/ollama serve > /dev/null 2>&1 &")
        prometheus = PrometheusHandler(ip_address, gpu_available)
        ollama = OllamaHandler(ip_address)
        prometheus.start_collection()
        ollama.process_models()
        prometheus.stop_collection()
        save_experiment()
       
    except (paramiko.AuthenticationException, paramiko.BadHostKeyException, paramiko.SSHException) as e:
        logger.exception_color(e)

    except Exception as e:
        logger.exception_color(e)
    
    finally:
        clean_local_resources()

        if ssh:
            clean_sut_resources(ssh)
            ssh.close()


if __name__ == '__main__':
    procesarLLM()