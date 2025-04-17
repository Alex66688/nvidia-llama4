# @alex66688/nvidia-llama4

Esta biblioteca permite integrar modelos de NVIDIA Llama4 con el ecosistema de LangChain.js, facilitando el uso de modelos como Llama4 de Meta en aplicaciones de procesamiento de lenguaje natural.

## Repositorio

[GitHub: https://github.com/Alex66688/nvidia-llama4](https://github.com/Alex66688/nvidia-llama4)

## Instalación

```bash
npm install @alex66688/nvidia-llama4 
yarn add @alex66688/nvidia-llama4 
pnpm add @alex66688/nvidia-llama4 
```

## Requisitos

- Node.js >= 18
- Una clave API de NVIDIA

## Modelos de Chat

```typescript
import { ChatNvidiaLlama4, ChatNvidiaLlama4Input, ChatNvidiaLlama4CallOptions } from "@alex66688/nvidia-llama4";

// Creación del modelo de chat
const chatModel = new ChatNvidiaLlama4({
  apiKey: "tu-api-key-de-nvidia",
  model: "meta/llama-4-maverick-17b-128e-instruct",
  temperature: 0.7,
  maxTokens: 512
});

// Generación de respuesta simple
const respuesta = await chatModel.invoke("Traduce 'Hola mundo' al inglés");

// Uso con mensajes estructurados
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
const messages = [
  new SystemMessage("Eres un asistente útil especializado en traducción"),
  new HumanMessage("Traduce 'Hola mundo' al inglés")
];
const chatResponse = await chatModel.invoke(messages);

// Streaming de respuestas
const stream = await chatModel.stream("Explica el concepto de inteligencia artificial");
for await (const chunk of stream) {
  console.log(chunk.content); // Muestra fragmentos de la respuesta conforme llegan
}
```

## Modelos de Lenguaje (LLMs)

```typescript
import { NvidiaLlama4, NvidiaLlama4Input, NvidiaLlama4CallOptions } from "@alex66688/nvidia-llama4";

// Creación del modelo de texto
const llm = new NvidiaLlama4({
  apiKey: "tu-api-key-de-nvidia",
  model: "meta/llama-4-maverick-17b-128e-instruct",
  temperature: 0.5,
  topP: 0.9
});

// Generación de texto
const respuesta = await llm.invoke("Explica qué es la inteligencia artificial");

// Uso con imágenes (para modelos multimodales)
const respuestaMultimodal = await llm.invoke("¿Qué muestra esta imagen?", {
  images: ["data:image/jpeg;base64,/9j/4AAQSkZJRg..."]  // Imagen en base64
});

// Streaming de respuestas
for await (const chunk of await llm.stream("Escribe un poema sobre la tecnología")) {
  process.stdout.write(chunk); // Imprime cada fragmento según llega
}
```

## Embeddings

```typescript
import { NvidiaEmbeddings, NvidiaEmbeddingsParams } from "@alex66688/nvidia-llama4";

// Creación del modelo de embeddings
const embeddings = new NvidiaEmbeddings({
  apiKey: "tu-api-key-de-nvidia",
  model: "embd/llama-4-embd"
});

// Generar embeddings para un texto individual
const vectorQuery = await embeddings.embedQuery("Este es un texto de ejemplo");

// Generar embeddings para múltiples textos
const documentos = [
  "La inteligencia artificial es una rama de la informática.",
  "El aprendizaje automático es un subcampo de la IA.",
  "Las redes neuronales son modelos inspirados en el cerebro humano."
];
const vectoresDocumentos = await embeddings.embedDocuments(documentos);
```

## Integración con LangChain

```typescript
import { ChatNvidiaLlama4 } from '@alex66688/nvidia-llama4';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

// Crear un modelo
const model = new ChatNvidiaLlama4({ 
  apiKey: "tu-api-key-de-nvidia",
  model: "meta/llama-4-maverick-17b-128e-instruct" 
});

// Crear una plantilla de prompt
const promptTemplate = PromptTemplate.fromTemplate(
  "Traduce el siguiente texto del español al {idioma}: {texto}"
);

// Crear una cadena (chain) de procesamiento
const chain = RunnableSequence.from([
  promptTemplate,
  model,
  new StringOutputParser()
]);

// Ejecutar la cadena
const resultado = await chain.invoke({
  idioma: "francés",
  texto: "Hola, ¿cómo estás?"
});
console.log(resultado);
```

## Documentación de API

Para más detalles sobre las clases, interfaces y métodos disponibles, consulta la documentación en el código fuente.

## Contribuir

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio en [https://github.com/Alex66688/nvidia-llama4](https://github.com/Alex66688/nvidia-llama4)
2. Crea una rama con tu nueva característica (`git checkout -b feature/amazing-feature`)
3. Haz commit de tus cambios (`git commit -m 'Añadir nueva característica'`)
4. Haz push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT.