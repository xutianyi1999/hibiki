use std::io::Write;
use anyhow::Result;
use llama_cpp_2 as llama;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn exec(decode_count: &AtomicU32, prompt_list: &[String]) -> Result<()> {
    let hf = hf_hub::api::sync::Api::new()?;
    let model_path = hf.model("TheBloke/openchat_3.5-GGUF".to_string())
        .get("openchat_3.5.Q5_K_M.gguf")?;

    let backend = llama::llama_backend::LlamaBackend::init()?;

    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(u32::MAX);

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

    let input_list = prompt_list.iter()
        .map(|prompt| (model.str_to_token(prompt, AddBos::Always).unwrap(), prompt))
        .collect::<Vec<_>>();

    std::thread::scope(|s| {
        for input_list in input_list.chunks(2) {
            s.spawn(|| {
                let batch_size = input_list.len() as i32;
                let ctx_params = LlamaContextParams::default().with_n_batch(batch_size as u32);
                let mut session = model.new_context(&backend, ctx_params)?;
                let ctx_size = session.n_ctx();
                let mut batch = LlamaBatch::new(512, batch_size);
                let mut sampler = LlamaSampler::chain_simple([
                    LlamaSampler::dist(3234),
                ]);

                // let input = model.apply_chat_template(Some("llama2".to_string()), vec![LlamaChatMessage::new("user".to_string(), "Hello!".to_string())?], false)?;

                let mut tokens_count = vec![0i32; input_list.len()];
                let mut curr_count = vec![0i32; input_list.len()];
                let mut out_text = Vec::new();

                for (i, (tokens, prompt)) in input_list.iter().enumerate() {
                    batch.add_sequence(tokens, i as i32, false)?;
                    tokens_count[i] = tokens.len() as i32;

                    let f= std::fs::File::options()
                        .write(true)
                        .create(true)
                        .open(format!("input_{}.txt", prompt))?;

                    out_text.push(std::io::BufWriter::new(f));
                }

                println!("before forward size: {}", session.get_state_size());

                loop {
                    session.decode(&mut batch)?;
                    batch.clear();

                    for seq_id in 0..batch_size as usize {
                        if tokens_count[seq_id] == -1 {
                            continue;
                        }

                        let out = sampler.sample(&session,-1);
                        decode_count.fetch_add(1, Ordering::Relaxed);

                        if model.is_eog_token(out) {
                            tokens_count[seq_id] = -1;
                            out_text[seq_id].flush()?;
                            println!("seq_id: {} eog", seq_id);
                            continue;
                        }

                        out_text[seq_id].write_all(model.token_to_str(out, Special::Plaintext)?.as_bytes())?;

                        batch.add(out, tokens_count[seq_id] - 1, &[seq_id as i32], true)?;
                        tokens_count[seq_id] += 1;
                        curr_count[seq_id] += 1;

                        if curr_count[seq_id] as u32 >= ctx_size {
                            println!("seq_id: {} clear_kv_cache", seq_id);
                            session.clear_kv_cache_seq(Some(seq_id as u32), Some(0), Some(ctx_size / 2))?;
                            session.kv_cache_seq_add(seq_id as i32, Some(ctx_size / 2), Some(ctx_size), ctx_size as i32 / 2)?;
                            curr_count[seq_id] /= 2;
                        }
                    }
                }

                println!("\nafter forward state size: {}", session.get_state_size());
                Ok::<(), anyhow::Error>(())
            });
        }
    });
    Ok(())
}

fn main() {
    let mut args = std::env::args();
    args.next();
    let prompt_list = args.next().unwrap();

    let prompt_list: Vec<String> = serde_json::from_str(&prompt_list).unwrap();
    let decode_count = Arc::new(AtomicU32::new(0));

    std::thread::spawn({
        let decode_count = decode_count.clone();

        move || {
            exec(decode_count.as_ref(), &prompt_list).unwrap();
        }
    });

    // wait llama.cpp setup
    std::thread::sleep(Duration::from_secs(5));
    decode_count.store(0, Ordering::Relaxed);

    loop {
        std::thread::sleep(Duration::from_secs(1));
        let count = decode_count.swap(0, Ordering::Relaxed);
        println!("decode count: {}", count);
    }
}
