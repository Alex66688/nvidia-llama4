import { NvidiaLlama4 } from "../src/llm";
import * as dotenv from "dotenv";

// Cargar variables de entorno desde .env
dotenv.config();

async function main() {
  // Verificar que la clave API está configurada
  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    console.error(
      "Error: Debes configurar la variable de entorno NVIDIA_API_KEY"
    );
    process.exit(1);
  }

  // Crear una instancia del modelo de lenguaje
  const llm = new NvidiaLlama4({
    apiKey,
    model: "meta/llama-4-maverick-17b-128e-instruct",
    temperature: 0.5,
    maxTokens: 1024,
    topP: 0.9,
  });

  console.log("Ejemplo 1: Generación de texto");
  console.log("-----------------------------");

  // Ejemplo 1: Generar texto simple
  const prompt =
    "Escribe un breve resumen sobre la inteligencia artificial y su impacto en la sociedad:";
  console.log(`Prompt: ${prompt}\n`);

  const respuesta = await llm.invoke(prompt);
  console.log(`Respuesta: ${respuesta}`);
  console.log("\n");

  console.log("Ejemplo 2: Streaming de respuestas");
  console.log("--------------------------------");

  // Ejemplo 2: Streaming de respuestas
  const promptStream =
    "Escribe tres consejos para programadores que están empezando:";
  console.log(`Prompt: ${promptStream}\n`);

  console.log("Generando respuesta por streaming...\n");
  process.stdout.write("Respuesta: ");

  const stream = await llm.stream(promptStream);
  for await (const chunk of stream) {
    process.stdout.write(chunk.toString());
  }

  console.log("\n\nStream completado.");
}

// Ejecutar el ejemplo
main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});