# Ejemplos de Uso de @rexdug7005/nvidia-llama4

Este directorio contiene ejemplos que muestran cómo utilizar la biblioteca @rexdug7005/nvidia-llama4 para interactuar con los modelos de NVIDIA LLaMA4.

## Requisitos Previos

Antes de ejecutar estos ejemplos, asegúrate de:

1. Tener Node.js instalado (v18 o superior recomendado)
2. Instalar las dependencias: `npm install`
3. Configurar tu clave API de NVIDIA en un archivo `.env` en la raíz del proyecto:

```
NVIDIA_API_KEY=tu_clave_api_aquí
```

## Ejecutar los Ejemplos

Para ejecutar cualquiera de los ejemplos, usa el siguiente comando:

```bash
# Usando npm
npm run build       # Primero compila el código
node dist/examples/nombre-del-ejemplo.js

# O usando ts-node para ejecutar directamente
npx ts-node examples/nombre-del-ejemplo.ts
```

## Descripción de los Ejemplos

### Chat Models (`chat-example.ts`)

Este ejemplo muestra cómo utilizar el modelo de chat `ChatNvidiaLlama4`:

- Generación de respuestas simples
- Uso de mensajes estructurados (SystemMessage, HumanMessage)
- Streaming de respuestas en tiempo real

### Language Models (`llm-example.ts`)

Este ejemplo muestra cómo utilizar el modelo de lenguaje `NvidiaLlama4`:

- Generación de texto completo
- Streaming de respuestas para obtener resultados en tiempo real

### Embeddings (`embeddings-example.ts`)

Este ejemplo muestra cómo utilizar `NvidiaEmbeddings`:

- Generación de embeddings para un solo texto
- Generación de embeddings para múltiples textos
- Cálculo de similitud de coseno entre vectores