import {
  AIMessage,
  BaseMessage,
  ChatMessage,
  MessageContentComplex,
} from "@langchain/core/messages";
import { z } from "zod";

/**
 * Opciones en formato camelCase para la configuración de modelos NVIDIA
 */
export interface NvidiaCamelCaseOptions {
  // Configuración general
  model?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;

  // Penalizaciones
  presencePenalty?: number;
  frequencyPenalty?: number;

  // Tokens de parada
  stop?: string[];

  // Opciones de imagen/multimodal
  images?: string[];
}

/**
 * Convierte opciones en formato camelCase a los parámetros esperados por la API de NVIDIA
 */
export function convertOptionsToNvidiaParams(
  options: NvidiaCamelCaseOptions
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  // Mapeo de nombres camelCase a los nombres de la API
  if (options.model !== undefined) result.model = options.model;
  if (options.maxTokens !== undefined) result.max_tokens = options.maxTokens;
  if (options.temperature !== undefined)
    result.temperature = options.temperature;
  if (options.topP !== undefined) result.top_p = options.topP;
  if (options.topK !== undefined) result.top_k = options.topK;
  if (options.presencePenalty !== undefined)
    result.presence_penalty = options.presencePenalty;
  if (options.frequencyPenalty !== undefined)
    result.frequency_penalty = options.frequencyPenalty;
  if (options.stop !== undefined) result.stop = options.stop;
  if (options.images !== undefined) result.images = options.images;

  return result;
}

/**
 * Definición del tipo para los mensajes en formato NVIDIA
 */
export const NvidiaMessageSchema = z.object({
  role: z.enum(["system", "user", "assistant"]),
  content: z.string().or(
    z.array(
      z.union([
        z.string(),
        z.object({
          type: z.literal("image"),
          image_url: z.object({
            url: z.string(),
          }),
        }),
      ])
    )
  ),
});

export type NvidiaMessage = z.infer<typeof NvidiaMessageSchema>;

/**
 * Formatea los mensajes de LangChain para la API de NVIDIA
 */
export function formatMessagesForNvidia(
  messages: BaseMessage[]
): NvidiaMessage[] {
  return messages.map((message): NvidiaMessage => {
    // Convertir de mensajes de LangChain a formato NVIDIA
    const messageType = message.constructor.name;

    if (messageType === "SystemMessage") {
      return {
        role: "system",
        content: message.content as string,
      };
    } else if (messageType === "HumanMessage") {
      // Manejar contenido multimodal para HumanMessage
      if (typeof message.content === "string") {
        return {
          role: "user",
          content: message.content,
        };
      } else {
        // Procesar contenido multimodal (texto + imagen)
        const content: (
          | string
          | { type: "image"; image_url: { url: string } }
        )[] = [];

        const parts = message.content as MessageContentComplex[];
        for (const part of parts) {
          if (part.type === "text") {
            content.push(part.text);
          } else if (part.type === "image_url") {
            content.push({
              type: "image",
              image_url: {
                url: part.image_url.url,
              },
            });
          }
        }

        return {
          role: "user",
          content,
        };
      }
    } else if (messageType === "AIMessage") {
      return {
        role: "assistant",
        content: message.content.toString(),
      };
    } else if (messageType === "ChatMessage") {
      // Mapear los roles de ChatMessage a los roles de NVIDIA
      let role: "system" | "user" | "assistant" = "user";
      const chatMessage = message as ChatMessage;

      if (chatMessage.role === "system") {
        role = "system";
      } else if (chatMessage.role === "assistant") {
        role = "assistant";
      } else {
        // Por defecto, asignar cualquier otro rol como "user"
        role = "user";
      }

      return {
        role,
        content: message.content as string,
      };
    } else {
      // Para cualquier otro tipo de mensaje, usar el rol de usuario
      return {
        role: "user",
        content: message.content.toString(),
      };
    }
  });
}

/**
 * Convierte la respuesta de NVIDIA a un mensaje de LangChain
 */
export function convertResponseToLangChainMessage(
  response: unknown
): AIMessage {
  // Extraer el contenido del mensaje de la respuesta
  const responseObj = response as {
    choices?: Array<{ message?: { content?: string }; finish_reason?: string }>;
    usage?: unknown;
  };
  const content = responseObj.choices?.[0]?.message?.content || "";

  // Crear un mensaje de IA con el contenido extraído
  return new AIMessage({
    content,
    // Opcional: Incluir metadatos adicionales si están disponibles
    additional_kwargs: {
      finish_reason: responseObj.choices?.[0]?.finish_reason,
      token_usage: responseObj.usage,
    },
  });
}