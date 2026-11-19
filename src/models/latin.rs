//! Repack the pinned INT8 reader's output head for English/Indonesian.
//!
//! The backbone is unchanged. Keep the CTC blank, Latin characters, numbers,
//! punctuation, symbols, Greek mathematical letters, and final space class.
//! This operates only on the SHA-256-verified parent in `models.rs`.

use anyhow::{Context, Result, ensure};
use rten_model_file::{
    header::Header,
    schema::{self, ConstantDataType, ConstantNode, Dim},
};

fn supported(character: char) -> bool {
    !character.is_alphabetic()
        || character.is_ascii()
        || matches!(character as u32,
            0x80..=0x2ff | 0x370..=0x3ff | 0x1e00..=0x1eff | 0x2c60..=0x2c7f
            | 0xa720..=0xa7ff | 0xab30..=0xab6f | 0xff00..=0xff5f | 0x1df00..=0x1dfff)
}

pub(super) fn derive(data: &[u8], dictionary: &str) -> Result<(Vec<u8>, String)> {
    let tokens: Vec<_> = dictionary.lines().collect();
    let mut keep = vec![0usize]; // CTC blank
    let mut new_dictionary = String::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.chars().all(supported) {
            keep.push(index + 1);
            new_dictionary.push_str(token);
            new_dictionary.push('\n');
        }
    }
    keep.push(tokens.len() + 1); // implicit space
    let old_classes = tokens.len() + 2;
    let new_classes = keep.len();
    ensure!(
        old_classes == 6906 && new_classes == 692,
        "unexpected reader alphabet"
    );

    let header = Header::from_buf(data)?;
    let model_start = header.model_offset as usize;
    let model_end = model_start + header.model_len as usize;
    let tensor_start = header.tensor_data_offset as usize;
    let model = schema::root_as_model(&data[model_start..model_end])?;
    let mut graph = data[..tensor_start].to_vec();
    let mut tensors = Vec::<u8>::new();
    let mut changed = 0;

    for node in model.graph().nodes().context("reader graph has no nodes")? {
        if let Some(constant) = node.data_as_constant_node() {
            let shape: Vec<_> = constant.shape().iter().map(|n| n as usize).collect();
            if let Some(offset) = constant.data_offset() {
                let width = match constant.dtype().context("tensor has no dtype")? {
                    ConstantDataType::Float32 | ConstantDataType::Int32 => 4,
                    ConstantDataType::Int8 | ConstantDataType::UInt8 => 1,
                    other => anyhow::bail!("unsupported reader tensor type {other:?}"),
                };
                let count = shape.iter().product::<usize>();
                let source_start = tensor_start + offset as usize;
                let source = &data[source_start..source_start + count * width];
                let block = if let Some(axis) = shape.iter().position(|&n| n == old_classes) {
                    ensure!(shape.iter().filter(|&&n| n == old_classes).count() == 1);
                    let prefix = shape[..axis].iter().product::<usize>();
                    let suffix = shape[axis + 1..].iter().product::<usize>();
                    let mut block = Vec::with_capacity(prefix * new_classes * suffix * width);
                    for pre in 0..prefix {
                        for &index in &keep {
                            let begin = (pre * old_classes + index) * suffix * width;
                            block.extend_from_slice(&source[begin..begin + suffix * width]);
                        }
                    }
                    let location = constant.shape().bytes().as_ptr() as usize
                        - data.as_ptr() as usize
                        + axis * 4;
                    graph[location..location + 4]
                        .copy_from_slice(&(new_classes as u32).to_le_bytes());
                    changed += 1;
                    block
                } else {
                    source.to_vec()
                };
                tensors.resize(tensors.len().next_multiple_of(64), 0);
                let field = constant._tab.vtable().get(ConstantNode::VT_DATA_OFFSET) as usize;
                ensure!(field != 0, "missing external tensor offset");
                let location = model_start + constant._tab.loc() + field;
                graph[location..location + 8]
                    .copy_from_slice(&(tensors.len() as u64).to_le_bytes());
                tensors.extend_from_slice(&block);
            } else {
                ensure!(
                    !shape.contains(&old_classes),
                    "unexpected embedded head tensor"
                );
            }
        }
        if let Some(value) = node.data_as_value_node()
            && let Some(shape) = value.shape()
        {
            for dimension in shape {
                if dimension.value() == old_classes as u32 {
                    let field = dimension._tab.vtable().get(Dim::VT_VALUE) as usize;
                    ensure!(field != 0, "missing output dimension");
                    let location = model_start + dimension._tab.loc() + field;
                    graph[location..location + 4]
                        .copy_from_slice(&(new_classes as u32).to_le_bytes());
                }
            }
        }
    }
    ensure!(
        changed == 4,
        "expected weight, bias, scale, and zero-point head tensors"
    );
    graph.extend(tensors);
    schema::root_as_model(&graph[model_start..model_end])?;
    Ok((graph, new_dictionary))
}
