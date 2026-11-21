# loopinfer-sdk

TypeScript SDK for the LoopInfer looped-transformer inference runtime.

```ts
import { LoopInferClient } from "loopinfer-sdk";

const client = new LoopInferClient("http://127.0.0.1:8080");
const res = await client.completions({ prompt: "循环层的等效深度", max_tokens: 32 });
console.log(res.choices[0].text);
```
