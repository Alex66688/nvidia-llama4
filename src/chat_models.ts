import {
  BaseChatModel,
  BaseChatModelCallOptions,
} from "@langchain/core/language_models/chat_models";
import { AIMessageChunk, BaseMessage } from "@langchain/core/messages";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { ChatGeneration, ChatGenerationChunk } from "@langchain/core/outputs";
import axios, { AxiosRequestConfig } from "axios";
import {
  NvidiaCamelCaseOptions,
  convertOptionsToNvidiaParams,
  convertResponseToLangChainMessage,
  formatMessagesForNvidia,
} from "./utils.js";

/**
 * Interfaz para las opciones de entrada del modelo de chat
 */
export interface ChatNvidiaLlama4Input {
  /** Clave API para NVIDIA Llama4 */
  apiKey: string;

  /** URL para las llamadas a la API. Por defecto es la URL de NVIDIA para chat */
  baseUrl?: string;

  /** El modelo a utilizar (por defecto: meta/llama-4-maverick-17b-128e-instruct) */
  model?: string;

  /** Habilitar o deshabilitar streaming */
  streaming?: boolean;

  /** Temperatura para la generación de texto (0-1) */
  temperature?: number;

  /** Número máximo de tokens a generar */
  maxTokens?: number;

  /** Valor de Top-P para muestreo de tokens */
  topP?: number;

  /** Valor de Top-K para muestreo de tokens */
  topK?: number;

  /** Penalización por presencia */
  presencePenalty?: number;

  /** Penalización por frecuencia */
  frequencyPenalty?: number;

  /** Tokens de parada */
  stop?: string[];
}

/**
 * Interfaz para las opciones de llamada del modelo de chat
 */
export interface ChatNvidiaLlama4CallOptions
  extends BaseChatModelCallOptions,
    NvidiaCamelCaseOptions {
  /** Lista de URLs de imágenes en formato base64 para entrada multimodal */
  images?: string[];
}

/**
 * Implementación del modelo de chat NVIDIA Llama4 para LangChain
 */
export class ChatNvidiaLlama4 extends BaseChatModel<ChatNvidiaLlama4CallOptions> {
  apiKey: string;

  baseUrl: string;

  modelName: string;

  defaultOptions: NvidiaCamelCaseOptions;

  streaming: boolean;

  static lc_name() {
    return "ChatNvidiaLlama4";
  }

  constructor(fields: ChatNvidiaLlama4Input) {
    // @ts-expect-error - Known issue with BaseLanguageModelParams
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
    messages: BaseMessage[],
    options: ChatNvidiaLlama4CallOptions,
    streaming: boolean = false
  ): Record<string, unknown> {
    // Convertir las opciones a formato NVIDIA
    const baseOptions = convertOptionsToNvidiaParams({
      ...this.defaultOptions,
      ...options,
      model: this.modelName,
    });

    // Formatear los mensajes para la API de NVIDIA
    const formattedMessages = formatMessagesForNvidia(messages);

    // Construir el payload final
    return {
      ...baseOptions,
      messages: formattedMessages,
      stream: streaming,
    };
  }

  /**
   * Genera una respuesta sincrónica (no streaming)
   */
  async _generate(
    messages: BaseMessage[],
    options: ChatNvidiaLlama4CallOptions
  ) {
    const requestOptions: AxiosRequestConfig = {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
        Accept: "application/json",
      },
    };

    const params = this.getParams(messages, options, false);

    try {
      const response = await axios.post(this.baseUrl, params, requestOptions);

      const responseData = response.data;
      const message = convertResponseToLangChainMessage(responseData);

      const generation: ChatGeneration = {
        text: message.content.toString(),
        message,
        generationInfo: {
          finishReason: responseData.choices?.[0]?.finish_reason,
          tokenUsage: responseData.usage,
        },
      };

      return {
        generations: [generation],
      };
    } catch (error: unknown) {
      throw new Error(
        `Error al llamar a la API de NVIDIA Llama4: ${String(error)}`
      );
    }
  }

  /**
   * Procesa la respuesta de streaming de la API
   */
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: ChatNvidiaLlama4CallOptions,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const requestOptions: AxiosRequestConfig = {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
        Accept: "text/event-stream",
      },
      responseType: "stream",
    };

    const params = this.getParams(messages, options, true);

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
              const content = parsedData.choices?.[0]?.delta?.content || "";

              if (content) {
                const messageChunk = new AIMessageChunk({
                  content,
                });

                const chunk = new ChatGenerationChunk({
                  text: content,
                  message: messageChunk,
                  generationInfo: {
                    finishReason: parsedData.choices?.[0]?.finish_reason,
                  },
                });

                yield chunk;

                // Notificar al manager de callbacks si existe
                if (runManager) {
                  await runManager.handleLLMNewToken(content);
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
   * Implementación del método _call requerido para ChatModels
   */
  async _call(
    messages: BaseMessage[],
    options: ChatNvidiaLlama4CallOptions
  ): Promise<string> {
    if (this.streaming) {
      let responseText = "";
      for await (const chunk of this._streamResponseChunks(
        messages,
        options
      )) {
        responseText += chunk.text;
      }
      return responseText;
    } else {
      const response = await this._generate(messages, options);
      if (!response.generations?.[0]?.text) {
        throw new Error("No se pudo generar texto con el modelo de chat");
      }
      return response.generations[0].text;
    }
  }
}