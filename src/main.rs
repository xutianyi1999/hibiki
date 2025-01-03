use std::io::Write;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use llama_cpp_2 as llama;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel, Special};
use llama_cpp_2::model::params::LlamaModelParams;
use anyhow::Result;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;

fn exec(batch_size: i32, decode_count: &AtomicU32) -> Result<()> {
    let hf = hf_hub::api::sync::Api::new()?;
    let model_path = hf.model("TheBloke/openchat_3.5-GGUF".to_string())
        .get("openchat_3.5.Q5_K_M.gguf")?;

    let backend = llama::llama_backend::LlamaBackend::init()?;

    let mut model_params = LlamaModelParams::default()
        .with_n_gpu_layers(u32::MAX);

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

    std::thread::scope(|s| {
        for _ in 0..2 {
            s.spawn(|| {
                let ctx_params = LlamaContextParams::default().with_n_batch(batch_size as u32);
                let mut session = model.new_context(&backend, ctx_params)?;

                let mut batch = LlamaBatch::new(512, batch_size);
                let mut sampler = LlamaSampler::chain_simple([
                    LlamaSampler::dist(3234),
                ]);

                // let input = model.apply_chat_template(Some("llama2".to_string()), vec![LlamaChatMessage::new("user".to_string(), "Hello!".to_string())?], false)?;
                let input = "Hello!";
                let tokens_list = model.str_to_token(&input, AddBos::Always)?;
                let tokens_len = tokens_list.len();
                let seq_ids = (0..=batch_size).collect::<Vec<i32>>();

                for (i, token) in tokens_list.into_iter().enumerate() {
                    batch.add(token, i as i32, &seq_ids, tokens_len - 1 == i)?;
                }

                println!("before forward size: {}", session.get_state_size());

                let mut n_cur = batch.n_tokens();
                println!("n_cur: {}", n_cur);

                loop {
                    session.decode(&mut batch)?;

                    for seq_id in 0..=batch_size {
                        let out = sampler.sample(&session,-1);
                        decode_count.fetch_add(1, Ordering::Relaxed);
                        if model.is_eog_token(out) {
                            continue;
                        }

                        // sampler.accept(out);

                        // let token_str = model.token_to_str(out, Special::Plaintext)?;
                        // print!("{}", token_str);
                        // std::io::stdout().flush()?;

                        batch.clear();
                        batch.add(out, n_cur, &[seq_id], true)?;
                    }

                    session.clear_kv_cache();
                    n_cur += 1;
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
    let batch_size_str = args.next().unwrap();
    let batch_size = i32::from_str(&batch_size_str).unwrap();

    let decode_count = Arc::new(AtomicU32::new(0));

    std::thread::spawn({
        let decode_count = decode_count.clone();

        move || {
            exec(batch_size, decode_count.as_ref()).unwrap();
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
