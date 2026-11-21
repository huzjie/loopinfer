import * as http from "http";
import { CompletionRequest, CompletionResponse } from "./types";

export class LoopInferClient {
  private baseUrl: string;

  constructor(baseUrl = "http://127.0.0.1:8080") {
    this.baseUrl = baseUrl;
  }

  completions(req: CompletionRequest): Promise<CompletionResponse> {
    return new Promise((resolve, reject) => {
      const body = JSON.stringify(req);
      const url = new URL("/v1/completions", this.baseUrl);
      const r = http.request(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Content-Length": Buffer.byteLength(body),
          },
        },
        (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () => {
            try {
              resolve(JSON.parse(data) as CompletionResponse);
            } catch (e) {
              reject(e);
            }
          });
        }
      );
      r.on("error", reject);
      r.write(body);
      r.end();
    });
  }

  health(): Promise<{ status: string }> {
    return new Promise((resolve, reject) => {
      const url = new URL("/health", this.baseUrl);
      http
        .get({ hostname: url.hostname, port: url.port, path: url.pathname }, (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () => resolve(JSON.parse(data)));
        })
        .on("error", reject);
    });
  }
}
