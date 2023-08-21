use std::io::Cursor;

use rand_0_8::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json;
use serde_json::to_string_pretty;
use zokrates_abi::{parse_strict, Decode, Encode, Inputs};
use zokrates_ark::Ark;
use zokrates_ast::ir;
use zokrates_ast::ir::ProgEnum;
pub use zokrates_ast::typed::abi::Abi;
use zokrates_bellman::Bellman;
pub use zokrates_common::helpers::{BackendParameter, SchemeParameter};
use zokrates_field::Field;
use zokrates_proof_systems::{
    groth16::G16, rng::get_rng_from_entropy, Backend, Marlin, Scheme, TaggedProof, GM17,
};

#[derive(thiserror::Error, Debug)]
pub enum HolonymError {
    #[error("Malformed")]
    Malformed,

    #[error("Invalid json value")]
    MalformedJSON,

    #[error("Invalid &[u8]")]
    DeserializationFailed,

    #[error("Interpreter Failed with error message {message}")]
    InterpreterFailed { message: String },

    #[error("Parse Failed with error message {message}")]
    ParseFailed { message: String },

    #[error("Witness reading failed with error message {message}")]
    WitnessReadFailed { message: String },

    #[error("Generate Proof with options: {options} not supported")]
    UnsupportedOption { options: GenerateProofOptions },

    #[error("IO error")]
    IoError,

    #[error("Invalid state")]
    InvalidState,

    #[error("Unsupported crypto or method")]
    Unsupported,

    #[error("Illegal argument")]
    IllegalArgument,
}

pub type Result<T> = std::result::Result<T, HolonymError>;

#[derive(Deserialize, Serialize, Debug)]
pub struct ComputationResult {
    witness: String,
    output: String,
}

impl UniffiCustomTypeConverter for Abi {
    type Builtin = String;

    fn into_custom(val: Self::Builtin) -> uniffi::Result<Self> {
        let value: Abi = serde_json::from_str(&val)?;
        Ok(value)
    }

    fn from_custom(obj: Self) -> Self::Builtin {
        serde_json::to_string(&obj).expect("unable serialize json value")
    }
}

impl UniffiCustomTypeConverter for ComputationResult {
    type Builtin = String;

    fn into_custom(val: Self::Builtin) -> uniffi::Result<Self> {
        let value: ComputationResult = serde_json::from_str(&val)?;
        Ok(value)
    }

    fn from_custom(obj: Self) -> Self::Builtin {
        serde_json::to_string(&obj).expect("unable serialize json value")
    }
}

pub fn compute<T: Field>(
    program: ir::Prog<T>,
    abi: Abi,
    input: String,
) -> Result<ComputationResult> {
    let signature = abi.signature();
    let inputs = parse_strict(&input, signature.inputs.clone())
        .map(Inputs::Abi)
        .map_err(|err| HolonymError::ParseFailed {
            message: format!("{err}",),
        })?;

    let interpreter = zokrates_interpreter::Interpreter::default();

    let witness = interpreter
        .execute(
            &inputs.encode(),
            program.statements.into_iter(),
            &program.arguments,
            &program.solvers,
        )
        .map_err(|err| match err {
            zokrates_interpreter::Error::LogStream => HolonymError::InterpreterFailed {
                message: format!("Interpreter Error writing a log to the log stream"),
            },
            zokrates_interpreter::Error::Solver(s) => HolonymError::InterpreterFailed {
                message: format!("Interpreter Solver error {}", s),
            },
            zokrates_interpreter::Error::WrongInputCount { expected, received } => {
                HolonymError::InterpreterFailed {
                    message: format!(
                        "Program takes {} input{} but was passed {} value{}",
                        expected,
                        if expected == 1 { "" } else { "s" },
                        received,
                        if received == 1 { "" } else { "s" }
                    ),
                }
            }
            zokrates_interpreter::Error::UnsatisfiedConstraint { error } => {
                HolonymError::InterpreterFailed {
                    message: format!(
                        "{}",
                        error
                            .as_ref()
                            .map(|m| m.to_string())
                            .expect("Found an unsatisfied constraint without an attached error.")
                    ),
                }
            }
        })?;

    let return_values: serde_json::Value =
        zokrates_abi::Value::decode(witness.return_values(), *signature.output).into_serde_json();

    Ok(ComputationResult {
        witness: format!("{}", witness),
        output: to_string_pretty(&return_values).unwrap(),
    })
}

pub fn compute_witness(program: &[u8], abi: Abi, args: String) -> Result<ComputationResult> {
    let cursor = Cursor::new(program);
    let prog = ir::ProgEnum::deserialize(cursor)
        .map_err(|_| HolonymError::DeserializationFailed)?
        .collect();

    match prog {
        ProgEnum::Bn128Program(p) => compute::<_>(p, abi, args),
        ProgEnum::Bls12_381Program(p) => compute::<_>(p, abi, args),
        ProgEnum::Bls12_377Program(p) => compute::<_>(p, abi, args),
        ProgEnum::Bw6_761Program(p) => compute::<_>(p, abi, args),
        ProgEnum::PallasProgram(p) => compute::<_>(p, abi, args),
        ProgEnum::VestaProgram(p) => compute::<_>(p, abi, args),
    }
}

impl UniffiCustomTypeConverter for BackendParameter {
    type Builtin = String;
    fn into_custom(val: Self::Builtin) -> uniffi::Result<Self> {
        let value: BackendParameter = serde_json::from_str(&val)?;
        Ok(value)
    }

    fn from_custom(obj: Self) -> Self::Builtin {
        serde_json::to_string(&obj).expect("unable serialize json value")
    }
}

impl UniffiCustomTypeConverter for SchemeParameter {
    type Builtin = String;
    fn into_custom(val: Self::Builtin) -> uniffi::Result<Self> {
        let value: SchemeParameter = serde_json::from_str(&val)?;
        Ok(value)
    }

    fn from_custom(obj: Self) -> Self::Builtin {
        serde_json::to_string(&obj).expect("unable serialize json value")
    }
}

#[derive(Deserialize, Serialize, Clone, Copy, Debug)]
pub struct GenerateProofOptions {
    backend: BackendParameter,
    scheme: SchemeParameter,
}

impl std::fmt::Display for GenerateProofOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.backend, self.scheme)
    }
}

fn generate<T: Field, S: Scheme<T>, B: Backend<T, S>, R: RngCore + CryptoRng>(
    prog: ir::Prog<T>,
    witness: String,
    pk: &[u8],
    rng: &mut R,
) -> Result<String> {
    let ir_witness: ir::Witness<T> =
        ir::Witness::read(witness.as_bytes()).map_err(|err| HolonymError::WitnessReadFailed {
            message: format!("{}", err),
        })?;
    let proof = B::generate_proof(prog, ir_witness, pk, rng);
    Ok(serde_json::to_string_pretty(&TaggedProof::<T, S>::new(proof.proof, proof.inputs)).unwrap())
}

pub fn generate_proof(
    program: &[u8],
    witness: String,
    pk: &[u8],
    entropy: String,
    options: GenerateProofOptions,
) -> Result<String> {
    let cursor = Cursor::new(program);
    let prog = ir::ProgEnum::deserialize(cursor)
        .map_err(|_| HolonymError::DeserializationFailed)?
        .collect();

    let mut rng = get_rng_from_entropy(&entropy);
    // TODO: maybe make entropy optional:
    // let mut rng = match entropy {
    //     Some(s) => get_rng_from_entropy(&s),
    //     None => { 
    //         rand_0_8::rngs::StdRng::from_entropy()
    //     }
    // }

    match (options.backend, options.scheme) {
        (BackendParameter::Bellman, SchemeParameter::G16) => match prog {
            ProgEnum::Bn128Program(p) => generate::<_, G16, Bellman, _>(p, witness, pk, &mut rng),
            // "Not supported: https://github.com/Zokrates/ZoKrates/issues/1200",
            ProgEnum::Bls12_381Program(_) => Err(HolonymError::UnsupportedOption { options }),
            _ => Err(HolonymError::UnsupportedOption { options }),
        },
        (BackendParameter::Ark, SchemeParameter::G16) => match prog {
            ProgEnum::Bn128Program(p) => generate::<_, G16, Ark, _>(p, witness, pk, &mut rng),
            ProgEnum::Bls12_381Program(p) => generate::<_, G16, Ark, _>(p, witness, pk, &mut rng),
            ProgEnum::Bls12_377Program(p) => generate::<_, G16, Ark, _>(p, witness, pk, &mut rng),
            ProgEnum::Bw6_761Program(p) => generate::<_, G16, Ark, _>(p, witness, pk, &mut rng),
            _ => Err(HolonymError::UnsupportedOption { options }),
        },
        (BackendParameter::Ark, SchemeParameter::GM17) => match prog {
            ProgEnum::Bn128Program(p) => generate::<_, GM17, Ark, _>(p, witness, pk, &mut rng),
            ProgEnum::Bls12_381Program(p) => generate::<_, GM17, Ark, _>(p, witness, pk, &mut rng),
            ProgEnum::Bls12_377Program(p) => generate::<_, GM17, Ark, _>(p, witness, pk, &mut rng),
            ProgEnum::Bw6_761Program(p) => generate::<_, GM17, Ark, _>(p, witness, pk, &mut rng),
            _ => Err(HolonymError::UnsupportedOption { options }),
        },
        (BackendParameter::Ark, SchemeParameter::MARLIN) => match prog {
            ProgEnum::Bn128Program(p) => generate::<_, Marlin, Ark, _>(p, witness, pk, &mut rng),
            ProgEnum::Bls12_381Program(p) => generate::<_, Marlin, Ark, _>(p, witness, pk, &mut rng),
            ProgEnum::Bls12_377Program(p) => generate::<_, Marlin, Ark, _>(p, witness, pk, &mut rng),
            ProgEnum::Bw6_761Program(p) => generate::<_, Marlin, Ark, _>(p, witness, pk, &mut rng),
            _ => Err(HolonymError::UnsupportedOption { options })
        },
        _ => Err(HolonymError::UnsupportedOption { options }),
    }
}

pub enum Circuit {
    OnAddLeaf,
    PoseidonTwoInputs,
    PoseidonQuinary,
    ProofOfResidency,
    AntiSybil,
    CreateLeaf,
}

include!(concat!(env!("OUT_DIR"), "/zokrates.uniffi.rs"));
