import { Embeddings, EmbeddingsParams } from "@langchain/core/embeddings";
import axios from "axios";

/**
 * Interfaz para los parámetros del modelo de embeddings
 */
export interface NvidiaEmbeddingsParams extends EmbeddingsParams {
  /** Clave API para NVIDIA Llama4 */
  apiKey: string;

  /** URL base para la API de embeddings */
  baseUrl?: string;

  /** Modelo de embeddings a usar */
  model?: string;

  /** Tipo de input (query o document) */
  inputType?: string;

  /** Formato de codificación (float) */
  encodingFormat?: string;

  /** Truncar texto (NONE, START, END) */
  truncate?: string;

  /** Número máximo de reintentos */
  maxRetries?: number;

  /** Temperatura para la generación de embeddings (0-1) */
  temperature?: number;
}

/**
 * Tipo de entrada para el modelo de embeddings
 */
export type NvidiaEmbeddingsInput = NvidiaEmbeddingsParams;

/**
 * Implementación de Embeddings de NVIDIA para LangChain
 */
export class NvidiaEmbeddings extends Embeddings {
  apiKey: string;

  baseUrl: string;

  modelName: string;

  inputType: string;

  encodingFormat: string;

  truncate: string;

  maxRetries: number;

  defaultOptions: Record<string, unknown>;

  constructor(fields: NvidiaEmbeddingsParams) {
    super(fields);

    this.apiKey = fields.apiKey;
    this.baseUrl =
      fields.baseUrl || "https://integrate.api.nvidia.com/v1/embeddings";
    this.modelName = fields.model || "nvidia/nv-embedcode-7b-v1";
    this.inputType = fields.inputType || "query";
    this.encodingFormat = fields.encodingFormat || "float";
    this.truncate = fields.truncate || "NONE";
    this.maxRetries = fields.maxRetries ?? 3;

    // Extraer las opciones que no son parte de la configuración principal
    const keysToExclude = [
      "apiKey",
      "baseUrl",
      "model",
      "inputType",
      "encodingFormat",
      "truncate",
      "maxRetries",
    ];
    // Creamos un objeto con todas las propiedades que no son de configuración principal
    this.defaultOptions = Object.fromEntries(
      Object.entries(fields).filter(([key]) => !keysToExclude.includes(key))
    );
  }

  /**
   * Método para realizar la llamada a la API con reintentos
   */
  private async embeddingWithRetry(
    text: string | string[]
  ): Promise<number[][]> {
    const texts = Array.isArray(text) ? text : [text];

    // Preparar payload para la API
    const payload = {
      model: this.modelName,
      input: texts,
      input_type: this.inputType,
      encoding_format: this.encodingFormat,
      truncate: this.truncate,
      ...this.defaultOptions,
    };

    // Opciones para la petición
    const requestOptions = {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
    };

    // Implementación de backoff exponencial para reintentos
    let error = "";
    for (let i = 0; i < this.maxRetries; i += 1) {
      try {
        const response = await axios.post(
          this.baseUrl,
          payload,
          requestOptions
        );

        return response.data.data.map(
          (item: { embedding: number[] }) => item.embedding
        );
      } catch (err: unknown) {
        error = String(err);
        // Esperar antes de reintentar (backoff exponencial)
        const waitTime = 2 ** i * 1000 + Math.random() * 100;
        // eslint-disable-next-line no-await-in-loop
        await new Promise((resolve) => {
          setTimeout(resolve, waitTime);
        });
      }
    }

    // Si llegamos aquí, todos los reintentos fallaron
    throw new Error(
      `Error al generar embeddings después de ${this.maxRetries} intentos: ${error}`
    );
  }

  /**
   * Generar embedding para un solo texto
   */
  async embedQuery(text: string): Promise<number[]> {
    const embeddings = await this.embeddingWithRetry(text);
    return embeddings[0];
  }

  /**
   * Generar embeddings para múltiples textos
   */
  async embedDocuments(documents: string[]): Promise<number[][]> {
    return this.embeddingWithRetry(documents);
  }
}