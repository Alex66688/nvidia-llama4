import {
  BaseLLM,
  BaseLLMCallOptions,
  BaseLLMParams,
} from "@langchain/core/language_models/llms";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { GenerationChunk } from "@langchain/core/outputs";
import axios, { AxiosRequestConfig } from "axios";
import {
  NvidiaCamelCaseOptions,
  convertOptionsToNvidiaParams,
} from "./utils.js";

/**
 * Interfaz para las opciones de entrada del modelo
 */
export interface NvidiaLlama4Input
  extends BaseLLMParams,
    NvidiaCamelCaseOptions {
  /** Clave API para NVIDIA Llama4 */
  apiKey: string;

  /** URL para las llamadas a la API */
  baseUrl?: string;

  /** El modelo a utilizar */
  model?: string;

  /** Habilitar streaming */
  streaming?: boolean;
}

/**
 * Interfaz para las opciones de llamada
 */
export interface NvidiaLlama4CallOptions
  extends BaseLLMCallOptions,
    NvidiaCamelCaseOptions {
  /** Lista de URLs de imágenes en formato base64 para entrada multimodal */
  images?: string[];
}

/**
 * Implementación del modelo de lenguaje NVIDIA Llama4 para LangChain
 */
export class NvidiaLlama4 extends BaseLLM<NvidiaLlama4CallOptions> {
  apiKey: string;

  baseUrl: string;

  modelName: string;

  defaultOptions: NvidiaCamelCaseOptions;

  streaming: boolean;

  static lc_name() {
    return "NvidiaLlama4";
  }

  constructor(fields: NvidiaLlama4Input) {
    super(fields);

    this.apiKey = fields.apiKey;
    this.baseUrl =
      fields.baseUrl || "https://integrate.api.nvidia.com/v1/chat/completions";
    this.modelName = fields.model || "meta/llama-4-maverick-17b-128e-instruct";
    this.streaming = fields.streaming ?? false;

    // Extraer opciones predeterminadas eliminando las propiedades que no son opciones del modelo
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { apiKey, baseUrl, model, streaming, ...rest } = fields;
    this.defaultOptions = rest;
  }

  _llmType() {
    return "nvidia-llama4";
  }

  /**
   * Obtiene los parámetros para la llamada a la API
   */
  private getParams(
    prompt: string,
    options: NvidiaLlama4CallOptions,
    streaming: boolean = false
  ): Record<string, unknown> {
    // Convertir las opciones a formato NVIDIA
    const baseOptions = convertOptionsToNvidiaParams({
      ...this.defaultOptions,
      ...options,
      model: this.modelName,
    });

    // Construir el payload para la API (formato de chat)
    const payload: Record<string, unknown> = {
      ...baseOptions,
      messages: [{ role: "user", content: prompt }],
      stream: streaming,
    };

    // Agregar imágenes si existen (para capacidades multimodales)
    if (options.images && options.images.length > 0) {
      payload.images = options.images;
    }

    return payload;
  }

  /**
   * Genera una respuesta sincrónica (no streaming)
   */
  async _generate(prompts: string[], options: NvidiaLlama4CallOptions) {
    const requestOptions: AxiosRequestConfig = {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
        Accept: "application/json",
      },
    };

    const generations = await Promise.all(
      prompts.map(async (prompt) => {
        const params = this.getParams(prompt, options, false);

        try {
          const response = await axios.post(
            this.baseUrl,
            params,
            requestOptions
          );

          const responseData = response.data;
          // En el formato de chat/completions, el texto está en choices[0].message.content
          const text = responseData.choices?.[0]?.message?.content || "";

          return [
            {
              text,
              generationInfo: {
                finishReason: responseData.choices?.[0]?.finish_reason,
                tokenUsage: responseData.usage,
              },
            },
          ];
        } catch (error: unknown) {
          throw new Error(
            `Error al llamar a la API de NVIDIA Llama4: ${String(error)}`
          );
        }
      })
    );

    return {
      generations,
    };
  }

  /**
   * Procesa la respuesta de streaming de la API
   */
  async *_streamResponseChunks(
    prompt: string,
    options: NvidiaLlama4CallOptions,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<GenerationChunk> {
    const requestOptions: AxiosRequestConfig = {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
        Accept: "text/event-stream",
      },
      responseType: "stream",
    };

    const params = this.getParams(prompt, options, true);

    try {
      const response = await axios.post(this.baseUrl, params, requestOptions);

      const stream = response.data;

      // Un buffer para acumular los datos del stream
      let buffer = "";

      for await (const chunk of stream) {
        const chunkText = Buffer.from(chunk).toString("utf-8");
        buffer += chunkText;

        // Procesar líneas completas
        while (buffer.includes("\n")) {
          const newlineIndex = buffer.indexOf("\n");
          const line = buffer.substring(0, newlineIndex).trim();
          buffer = buffer.substring(newlineIndex + 1);

          if (line.startsWith("data: ")) {
            const data = line.substring(6).trim();

            // Fin del stream
            if (data === "[DONE]") {
              return;
            }

            try {
              const parsedData = JSON.parse(data);
              // En el formato de chat/completions, el contenido está en choices[0].delta.content
              const text = parsedData.choices?.[0]?.delta?.content || "";

              if (text) {
                const chunk = new GenerationChunk({
                  text,
                  generationInfo: {
                    finishReason: parsedData.choices?.[0]?.finish_reason,
                  },
                });

                yield chunk;

                // Notificar al manager de callbacks si existe
                if (runManager) {
                  await runManager.handleLLMNewToken(text);
                }
              }
            } catch (error) {
              // Ignorar líneas no válidas
              continue;
            }
          }
        }
      }
    } catch (error: unknown) {
      throw new Error(
        `Error al procesar el stream de NVIDIA Llama4: ${String(error)}`
      );
    }
  }

  /**
   * Implementación del método _call requerido para LLMs
   */
  async _call(
    prompt: string,
    options: NvidiaLlama4CallOptions
  ): Promise<string> {
    if (this.streaming) {
      let responseText = "";
      for await (const chunk of this._streamResponseChunks(prompt, options)) {
        if (chunk && chunk.text) {
          responseText += chunk.text;
        }
      }
      return responseText;
    } else {
      const response = await this._generate([prompt], options);
      if (!response.generations?.[0]?.[0]?.text) {
        throw new Error("No se pudo generar texto con el modelo");
      }
      return response.generations[0][0].text;
    }
  }
}