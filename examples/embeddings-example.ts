import { NvidiaEmbeddings } from "../src/embeddings";
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

  // Crear una instancia del modelo de embeddings
  const embeddings = new NvidiaEmbeddings({
    apiKey,
    model: "nvidia/nv-embedcode-7b-v1",
    inputType: "query",
    encodingFormat: "float",
    truncate: "NONE",
  });

  console.log("Ejemplo 1: Generar embeddings para un texto");
  console.log("------------------------------------------");

  // Generar embedding para un texto individual
  const texto =
    "La inteligencia artificial está transformando la forma en que interactuamos con la tecnología.";
  console.log(`Texto: "${texto}"\n`);

  console.log("Generando embedding...");
  const embedding = await embeddings.embedQuery(texto);

  // Mostrar dimensiones y una muestra del vector
  console.log(`\nDimensiones del vector: ${embedding.length}`);
  console.log(`Primeros 5 valores: ${embedding.slice(0, 5).join(", ")}`);
  console.log(`Últimos 5 valores: ${embedding.slice(-5).join(", ")}`);
  console.log("\n");

  console.log("Ejemplo 2: Generar embeddings para múltiples textos");
  console.log("--------------------------------------------------");

  // Generar embeddings para varios textos
  const textos = [
    "La inteligencia artificial es una rama de la informática.",
    "El aprendizaje automático es un subcampo de la IA.",
    "Las redes neuronales están inspiradas en el cerebro humano.",
  ];

  console.log("Textos:");
  textos.forEach((t, i) => console.log(`${i + 1}. "${t}"`));
  console.log("\nGenerando embeddings para los textos...");

  const embeddingsResultado = await embeddings.embedDocuments(textos);

  // Mostrar información sobre los embeddings generados
  console.log(
    `\nNúmero de embeddings generados: ${embeddingsResultado.length}`
  );
  console.log(`Dimensiones de cada vector: ${embeddingsResultado[0].length}`);

  // Calcular similitud coseno entre los dos primeros vectores (como ejemplo)
  if (embeddingsResultado.length >= 2) {
    const similaridad = calcularSimilitudCoseno(
      embeddingsResultado[0],
      embeddingsResultado[1]
    );
    console.log(
      `\nSimilitud coseno entre texto 1 y texto 2: ${similaridad.toFixed(4)}`
    );
  }
}

// Función auxiliar para calcular la similitud coseno entre dos vectores
function calcularSimilitudCoseno(a: number[], b: number[]): number {
  // Producto escalar
  const productoEscalar = a.reduce((sum, aVal, i) => sum + aVal * b[i], 0);

  // Magnitudes
  const magnitudA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

  // Similitud coseno
  return productoEscalar / (magnitudA * magnitudB);
}

// Ejecutar el ejemplo
main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});