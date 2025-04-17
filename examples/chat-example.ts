import { ChatNvidiaLlama4 } from "../src/chat_models";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import * as dotenv from "dotenv";

// Cargar variables de entorno desde .env
dotenv.config();

async function main() {
  // Verificar que la clave API está configurada
  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    console.error("Error: Debes configurar la variable de entorno NVIDIA_API_KEY");
    process.exit(1);
  }

  // Crear una instancia del modelo de chat
  const chatModel = new ChatNvidiaLlama4({
    apiKey,
    model: "meta/llama-4-maverick-17b-128e-instruct",
    temperature: 0.7,
    maxTokens: 1024
  });

  console.log("Ejemplo 1: Respuesta simple");
  console.log("---------------------------");
  
  // Ejemplo 1: Generar una respuesta simple
  const respuestaSimple = await chatModel.invoke("Explica de manera sencilla qué es la inteligencia artificial");
  console.log(respuestaSimple.content);
  console.log("\n");

  console.log("Ejemplo 2: Mensajes estructurados");
  console.log("--------------------------------");
  
  // Ejemplo 2: Usar mensajes estructurados
  const messages = [
    new SystemMessage("Eres un profesor de español que ayuda a los estudiantes a aprender español."),
    new HumanMessage("¿Puedes explicarme la diferencia entre 'ser' y 'estar' en español?")
  ];
  
  const respuestaEstructurada = await chatModel.invoke(messages);
  console.log(respuestaEstructurada.content);
  console.log("\n");

  console.log("Ejemplo 3: Streaming de respuestas");
  console.log("--------------------------------");
  
  // Ejemplo 3: Streaming de respuestas
  console.log("Generando respuesta por streaming...\n");
  
  const stream = await chatModel.stream("Escribe un poema corto sobre la tecnología y la naturaleza");
  
  process.stdout.write("Respuesta: ");
  for await (const chunk of stream) {
    // Convertir el contenido a string antes de escribirlo
    const content = String(chunk.content);
    process.stdout.write(content);
  }
  console.log("\n\nStream completado.");
}

// Ejecutar el ejemplo
main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});