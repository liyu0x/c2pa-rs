use std::{
    collections::BTreeMap,
    env,
    fs::File,
    hint::black_box,
    io::{Cursor, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use blst::{min_pk as bls, BLST_ERROR};
use c2pa::{
    build_aggregate_certificate_profile, build_receipt_bound_export_context_digest,
    canonicalize_roster, export_content_hash_from_stream, issue_platform_acceptance_receipt,
    AggregateCertificateProfile, AggregatePublicKey, AggregateSignature, AuthorizationPackage,
    Builder, CallbackSigner, CollaborativeVerifier, CollectiveSigner,
    EmbeddedCollaborativeManifest, Error, ExportContextDigest, IssuedAggregateCertificate,
    IssuedPlatformReceiptCertificate, PartialSignature, Participant, Reader, Result, SigningAlg,
    TrustStore, ALLOWED_CUSTOM_EXTENSION_OIDS, COLLABORATIVE_AUTHORIZATION_LABEL,
};
use openssl::{
    asn1::{Asn1Object, Asn1OctetString, Asn1Time},
    bn::BigNum,
    hash::MessageDigest,
    pkey::{PKey, Private},
    rsa::Rsa,
    x509::{
        extension::{BasicConstraints, KeyUsage},
        X509Extension, X509NameBuilder, X509,
    },
};
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};

const CERTS: &[u8] = include_bytes!("../tests/fixtures/certs/ed25519.pub");
const PRIVATE_KEY: &[u8] = include_bytes!("../tests/fixtures/certs/ed25519.pem");
const SOURCE_JPEG: &[u8] = include_bytes!("../tests/fixtures/IMG_0003.jpg");
const MANIFEST_JSON: &str = include_str!("../tests/fixtures/simple_manifest.json");
const PARTICIPANT_COUNTS: &[usize] = &[20, 50, 100];
const BLS_ALGORITHM: &str = "BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
const BLS_SIGNATURE_DST: &[u8] = BLS_ALGORITHM.as_bytes();
const BLS_POP_DST: &[u8] = b"BLS_POP_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

#[derive(Clone)]
struct BlsCollectiveSigner {
    secret_keys: BTreeMap<String, Vec<u8>>,
}

impl CollectiveSigner for BlsCollectiveSigner {
    fn algorithm(&self) -> &'static str {
        BLS_ALGORITHM
    }

    fn aggregate_public_key(&self, roster: &[Participant]) -> Result<AggregatePublicKey> {
        let canonical = canonicalize_roster(roster)?;
        let public_keys = canonical
            .iter()
            .map(|participant| {
                bls::PublicKey::from_bytes(&participant.public_key_der).map_err(bls_error)
            })
            .collect::<Result<Vec<_>>>()?;
        let public_key_refs = public_keys.iter().collect::<Vec<_>>();
        let aggregate = bls::AggregatePublicKey::aggregate(&public_key_refs, true)
            .map_err(bls_error)?
            .to_public_key();
        Ok(AggregatePublicKey {
            algorithm: self.algorithm().to_owned(),
            bytes: aggregate.to_bytes().to_vec(),
        })
    }

    fn partial_sign(
        &self,
        participant: &Participant,
        export_context_digest: &ExportContextDigest,
    ) -> Result<PartialSignature> {
        let secret_key_bytes = self
            .secret_keys
            .get(&participant.identifier)
            .ok_or_else(|| {
                io_error(
                    std::io::ErrorKind::PermissionDenied,
                    "missing BLS secret key for participant",
                )
            })?;
        let secret_key = bls::SecretKey::from_bytes(secret_key_bytes).map_err(bls_error)?;
        let signature = secret_key.sign(&export_context_digest.0, BLS_SIGNATURE_DST, &[]);
        Ok(PartialSignature {
            participant_id: participant.identifier.clone(),
            bytes: signature.to_bytes().to_vec(),
        })
    }

    fn aggregate_signatures(
        &self,
        _export_context_digest: &ExportContextDigest,
        roster: &[Participant],
        partials: &[PartialSignature],
    ) -> Result<AggregateSignature> {
        let canonical = canonicalize_roster(roster)?;
        let mut signatures = Vec::with_capacity(canonical.len());
        for participant in canonical {
            let partial = partials
                .iter()
                .find(|partial| partial.participant_id == participant.identifier)
                .ok_or_else(|| {
                    io_error(
                        std::io::ErrorKind::InvalidInput,
                        "missing partial signature",
                    )
                })?;
            signatures.push(bls::Signature::from_bytes(&partial.bytes).map_err(bls_error)?);
        }
        let signature_refs = signatures.iter().collect::<Vec<_>>();
        let aggregate = bls::AggregateSignature::aggregate(&signature_refs, true)
            .map_err(bls_error)?
            .to_signature();
        Ok(AggregateSignature {
            algorithm: self.algorithm().to_owned(),
            bytes: aggregate.to_bytes().to_vec(),
        })
    }

    fn verify(
        &self,
        export_context_digest: &ExportContextDigest,
        aggregate_public_key: &AggregatePublicKey,
        signature: &AggregateSignature,
        roster: &[Participant],
    ) -> Result<()> {
        let expected_apk = self.aggregate_public_key(roster)?;
        if expected_apk != *aggregate_public_key {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                "aggregate public key mismatch",
            ));
        }
        if signature.algorithm != self.algorithm() {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                "unexpected aggregate signature algorithm",
            ));
        }
        let canonical = canonicalize_roster(roster)?;
        let public_keys = canonical
            .iter()
            .map(|participant| {
                bls::PublicKey::from_bytes(&participant.public_key_der).map_err(bls_error)
            })
            .collect::<Result<Vec<_>>>()?;
        let public_key_refs = public_keys.iter().collect::<Vec<_>>();
        let aggregate_signature =
            bls::Signature::from_bytes(&signature.bytes).map_err(bls_error)?;
        if aggregate_signature.fast_aggregate_verify(
            true,
            &export_context_digest.0,
            BLS_SIGNATURE_DST,
            &public_key_refs,
        ) != BLST_ERROR::BLST_SUCCESS
        {
            return Err(io_error(
                std::io::ErrorKind::InvalidData,
                "BLS aggregate signature verification failed",
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct PreparedEnrollment {
    roster: Vec<Participant>,
    collective_signer: BlsCollectiveSigner,
    issued_certificate: IssuedAggregateCertificate,
    platform_private_key_pem: Vec<u8>,
    platform_receipt_certificate: IssuedPlatformReceiptCertificate,
    root_pem: String,
}

struct BenchmarkFixture {
    enrollments: Vec<BlsEnrollment>,
    issuer: TestIssuer,
}

struct BlsEnrollment {
    participant: Participant,
    secret_key: Vec<u8>,
    proof_of_possession: Vec<u8>,
}

struct TestIssuer {
    root_key: PKey<Private>,
    root_cert: X509,
    leaf_key: PKey<Private>,
    root_pem: String,
}

#[derive(Clone)]
struct Sample {
    scenario: &'static str,
    participants: usize,
    run: usize,
    enroll_ms: f64,
    finalize_ms: f64,
    verify_ms: f64,
    file_sign_ms: f64,
    file_verify_ms: f64,
    standard_sign_ms: f64,
    standard_read_ms: f64,
    total_ms: f64,
    output_bytes: usize,
}

struct Config {
    runs: usize,
    warmups: usize,
    csv_path: Option<PathBuf>,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let mut samples = Vec::new();

    eprintln!(
        "Running collaborative benchmark: runs={}, warmups={}, jpeg_bytes={}",
        config.runs,
        config.warmups,
        SOURCE_JPEG.len()
    );

    let fixtures = PARTICIPANT_COUNTS
        .iter()
        .map(|participants| {
            eprintln!("building fixture for n={participants}");
            Ok((*participants, build_fixture(*participants)?))
        })
        .collect::<Result<Vec<_>>>()?;

    for warmup in 0..config.warmups {
        eprintln!("warmup {}", warmup + 1);
        black_box(run_standard_once(0)?);
        for (participants, fixture) in &fixtures {
            black_box(run_protocol_once(*participants, fixture, 0)?);
            black_box(run_file_chain_once(*participants, fixture, 0)?);
        }
    }

    for run in 1..=config.runs {
        eprintln!("recorded run {run}");
        samples.push(run_standard_once(run)?);
        for (participants, fixture) in &fixtures {
            samples.push(run_protocol_once(*participants, fixture, run)?);
            samples.push(run_file_chain_once(*participants, fixture, run)?);
        }
    }

    if let Some(path) = &config.csv_path {
        let mut file = File::create(path)?;
        write_samples_csv(&mut file, &samples)?;
        eprintln!("raw csv: {}", path.display());
    } else {
        let mut stdout = std::io::stdout().lock();
        write_samples_csv(&mut stdout, &samples)?;
    }

    print_summary(&samples);
    Ok(())
}

fn parse_args() -> std::result::Result<Config, Box<dyn std::error::Error>> {
    let mut runs = 30;
    let mut warmups = 1;
    let mut csv_path = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--runs" => {
                runs = args
                    .next()
                    .ok_or("--runs requires a value")?
                    .parse::<usize>()?;
            }
            "--warmups" => {
                warmups = args
                    .next()
                    .ok_or("--warmups requires a value")?
                    .parse::<usize>()?;
            }
            "--csv" => {
                csv_path = Some(PathBuf::from(args.next().ok_or("--csv requires a path")?));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example collaborative_benchmark -- [--runs N] [--warmups N] [--csv PATH]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Config {
        runs,
        warmups,
        csv_path,
    })
}

fn run_protocol_once(
    participants: usize,
    fixture: &BenchmarkFixture,
    run: usize,
) -> Result<Sample> {
    let total_start = Instant::now();

    let enroll_start = Instant::now();
    let prepared = prepare_enrollment(fixture)?;
    let enroll_ms = elapsed_ms(enroll_start.elapsed());

    let finalize_start = Instant::now();
    let embedded = finalize_authorization(&prepared, run)?;
    let finalize_ms = elapsed_ms(finalize_start.elapsed());

    let verify_start = Instant::now();
    let export_content_hash = export_content_hash_from_stream("image/jpeg", SOURCE_JPEG)?;
    verify_embedded(&embedded, &prepared, &export_content_hash)?;
    let verify_ms = elapsed_ms(verify_start.elapsed());

    Ok(Sample {
        scenario: "protocol",
        participants,
        run,
        enroll_ms,
        finalize_ms,
        verify_ms,
        file_sign_ms: 0.0,
        file_verify_ms: 0.0,
        standard_sign_ms: 0.0,
        standard_read_ms: 0.0,
        total_ms: elapsed_ms(total_start.elapsed()),
        output_bytes: 0,
    })
}

fn run_file_chain_once(
    participants: usize,
    fixture: &BenchmarkFixture,
    run: usize,
) -> Result<Sample> {
    let total_start = Instant::now();

    let enroll_start = Instant::now();
    let prepared = prepare_enrollment(fixture)?;
    let enroll_ms = elapsed_ms(enroll_start.elapsed());

    let sign_start = Instant::now();
    let embedded = finalize_authorization(&prepared, run)?;
    let signed_bytes = sign_collaborative_jpeg(&embedded)?;
    let file_sign_ms = elapsed_ms(sign_start.elapsed());

    let verify_start = Instant::now();
    verify_signed_collaborative_jpeg(&signed_bytes, &prepared)?;
    let file_verify_ms = elapsed_ms(verify_start.elapsed());

    Ok(Sample {
        scenario: "file_chain",
        participants,
        run,
        enroll_ms,
        finalize_ms: 0.0,
        verify_ms: 0.0,
        file_sign_ms,
        file_verify_ms,
        standard_sign_ms: 0.0,
        standard_read_ms: 0.0,
        total_ms: elapsed_ms(total_start.elapsed()),
        output_bytes: signed_bytes.len(),
    })
}

fn run_standard_once(run: usize) -> Result<Sample> {
    let total_start = Instant::now();

    let sign_start = Instant::now();
    let signed_bytes = sign_standard_jpeg()?;
    let standard_sign_ms = elapsed_ms(sign_start.elapsed());

    let read_start = Instant::now();
    read_standard_jpeg(&signed_bytes)?;
    let standard_read_ms = elapsed_ms(read_start.elapsed());

    Ok(Sample {
        scenario: "standard_c2pa",
        participants: 0,
        run,
        enroll_ms: 0.0,
        finalize_ms: 0.0,
        verify_ms: 0.0,
        file_sign_ms: 0.0,
        file_verify_ms: 0.0,
        standard_sign_ms,
        standard_read_ms,
        total_ms: elapsed_ms(total_start.elapsed()),
        output_bytes: signed_bytes.len(),
    })
}

fn build_fixture(participants: usize) -> Result<BenchmarkFixture> {
    let enrollments = (0..participants)
        .map(|index| {
            let participant_id = format!("participant-{index:03}");
            let nonce = format!("fixture-nonce-{participants}-{index}");
            make_bls_enrollment(&participant_id, nonce.as_bytes())
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(BenchmarkFixture {
        enrollments,
        issuer: make_test_issuer()?,
    })
}

fn prepare_enrollment(fixture: &BenchmarkFixture) -> Result<PreparedEnrollment> {
    for enrollment in &fixture.enrollments {
        verify_bls_proof_of_possession(enrollment)?;
    }
    let roster = canonicalize_roster(
        &fixture
            .enrollments
            .iter()
            .map(|enrollment| enrollment.participant.clone())
            .collect::<Vec<_>>(),
    )?;
    let signer = BlsCollectiveSigner {
        secret_keys: fixture
            .enrollments
            .iter()
            .map(|enrollment| {
                (
                    enrollment.participant.identifier.clone(),
                    enrollment.secret_key.clone(),
                )
            })
            .collect(),
    };
    let aggregate_public_key = signer.aggregate_public_key(&roster)?;
    let profile = build_aggregate_certificate_profile(aggregate_public_key, &roster)?;
    let issued_certificate = issue_test_collaborative_certificate(&profile, &fixture.issuer)?;

    Ok(PreparedEnrollment {
        roster,
        collective_signer: signer,
        platform_private_key_pem: fixture
            .issuer
            .leaf_key
            .private_key_to_pem_pkcs8()
            .map_err(other_error)?,
        platform_receipt_certificate: IssuedPlatformReceiptCertificate {
            leaf_certificate_pem: issued_certificate.leaf_certificate_pem.clone(),
            issuer_certificate_pem: issued_certificate.issuer_certificate_pem.clone(),
        },
        issued_certificate,
        root_pem: fixture.issuer.root_pem.clone(),
    })
}

fn finalize_authorization(
    prepared: &PreparedEnrollment,
    run: usize,
) -> Result<EmbeddedCollaborativeManifest> {
    let signer = &prepared.collective_signer;
    let session_id = format!("benchmark-session-{run}");
    let final_state_hash =
        Sha256::digest(format!("benchmark-final-state-{run}").as_bytes()).to_vec();
    let export_content_hash = export_content_hash_from_stream("image/jpeg", SOURCE_JPEG)?;
    let platform_acceptance_receipts = prepared
        .roster
        .iter()
        .enumerate()
        .map(|(index, participant)| {
            issue_platform_acceptance_receipt(
                &prepared.platform_private_key_pem,
                &session_id,
                participant,
                format!("benchmark-accepted-action-{run}-{index}").as_bytes(),
                index as u64,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let final_export_context_digest = build_receipt_bound_export_context_digest(
        &session_id,
        &final_state_hash,
        &export_content_hash,
        &prepared.roster,
        &platform_acceptance_receipts,
    )?;
    let partials = prepared
        .roster
        .iter()
        .map(|participant| signer.partial_sign(participant, &final_export_context_digest))
        .collect::<Result<Vec<_>>>()?;
    let aggregate_signature =
        signer.aggregate_signatures(&final_export_context_digest, &prepared.roster, &partials)?;

    let authorization = AuthorizationPackage {
        session_id,
        final_state_hash,
        export_content_hash,
        final_export_context_digest,
        platform_acceptance_receipts,
        platform_receipt_certificate: prepared.platform_receipt_certificate.clone(),
        aggregate_signature,
        aggregate_certificate_hash: hex::encode(Sha256::digest(serde_json::to_vec(
            &prepared.issued_certificate,
        )?)),
    };

    Ok(EmbeddedCollaborativeManifest {
        authorization,
        aggregate_certificate: prepared.issued_certificate.clone(),
    })
}

fn verify_embedded(
    embedded: &EmbeddedCollaborativeManifest,
    prepared: &PreparedEnrollment,
    expected_export_content_hash: &[u8],
) -> Result<()> {
    let verifier = CollaborativeVerifier::with_trust_store(
        prepared.collective_signer.clone(),
        TrustStore::from_pem(prepared.root_pem.clone())?,
    );
    black_box(verifier.verify_embedded(
        embedded,
        &prepared.roster,
        expected_export_content_hash,
    )?);
    Ok(())
}

fn sign_collaborative_jpeg(embedded: &EmbeddedCollaborativeManifest) -> Result<Vec<u8>> {
    let file_signer = make_file_signer();
    let mut builder = Builder::default().with_definition(MANIFEST_JSON)?;
    builder.add_assertion_json(COLLABORATIVE_AUTHORIZATION_LABEL, embedded)?;

    let mut source = Cursor::new(SOURCE_JPEG);
    let mut dest = Cursor::new(Vec::new());
    black_box(builder.sign(&file_signer, "image/jpeg", &mut source, &mut dest)?);
    Ok(dest.into_inner())
}

fn verify_signed_collaborative_jpeg(
    signed_bytes: &[u8],
    prepared: &PreparedEnrollment,
) -> Result<()> {
    let mut stream = Cursor::new(signed_bytes);
    let reader = Reader::default().with_stream("image/jpeg", &mut stream)?;
    let active_manifest = reader
        .active_manifest()
        .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "missing active manifest"))?;
    let embedded: EmbeddedCollaborativeManifest =
        active_manifest.find_assertion(COLLABORATIVE_AUTHORIZATION_LABEL)?;
    let export_content_hash = export_content_hash_from_stream("image/jpeg", signed_bytes)?;
    verify_embedded(&embedded, prepared, &export_content_hash)
}

fn sign_standard_jpeg() -> Result<Vec<u8>> {
    let file_signer = make_file_signer();
    let mut builder = Builder::default().with_definition(MANIFEST_JSON)?;
    let mut source = Cursor::new(SOURCE_JPEG);
    let mut dest = Cursor::new(Vec::new());
    black_box(builder.sign(&file_signer, "image/jpeg", &mut source, &mut dest)?);
    Ok(dest.into_inner())
}

fn read_standard_jpeg(signed_bytes: &[u8]) -> Result<()> {
    let mut stream = Cursor::new(signed_bytes);
    let reader = Reader::default().with_stream("image/jpeg", &mut stream)?;
    if reader.active_manifest().is_none() {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            "missing active manifest",
        ));
    }
    black_box(reader.json());
    Ok(())
}

fn make_file_signer() -> CallbackSigner {
    let ed_signer =
        |_context: *const (), data: &[u8]| CallbackSigner::ed25519_sign(data, PRIVATE_KEY);
    CallbackSigner::new(ed_signer, SigningAlg::Ed25519, CERTS)
}

fn make_bls_enrollment(participant_id: &str, nonce: &[u8]) -> Result<BlsEnrollment> {
    let mut ikm = [0u8; 32];
    OsRng.fill_bytes(&mut ikm);
    let secret_key = bls::SecretKey::key_gen(&ikm, nonce).map_err(bls_error)?;
    let public_key = secret_key.sk_to_pk();
    let public_key_bytes = public_key.to_bytes().to_vec();
    let participant = Participant {
        identifier: participant_id.to_string(),
        certificate_fingerprint: hex::encode(Sha256::digest(&public_key_bytes)),
        public_key_der: public_key_bytes.clone(),
    };
    let proof_of_possession = secret_key.sign(&public_key_bytes, BLS_POP_DST, &[]);
    Ok(BlsEnrollment {
        participant,
        secret_key: secret_key.to_bytes().to_vec(),
        proof_of_possession: proof_of_possession.to_bytes().to_vec(),
    })
}

fn verify_bls_proof_of_possession(enrollment: &BlsEnrollment) -> Result<()> {
    let public_key =
        bls::PublicKey::from_bytes(&enrollment.participant.public_key_der).map_err(bls_error)?;
    let proof = bls::Signature::from_bytes(&enrollment.proof_of_possession).map_err(bls_error)?;
    if proof.verify(
        true,
        &enrollment.participant.public_key_der,
        BLS_POP_DST,
        &[],
        &public_key,
        true,
    ) != BLST_ERROR::BLST_SUCCESS
    {
        return Err(io_error(
            std::io::ErrorKind::PermissionDenied,
            "BLS proof of possession verification failed",
        ));
    }
    Ok(())
}

fn bls_error(error: BLST_ERROR) -> Error {
    io_error(
        std::io::ErrorKind::InvalidData,
        format!("BLS12-381 operation failed: {error:?}"),
    )
}

fn make_test_issuer() -> Result<TestIssuer> {
    let root_key =
        PKey::from_rsa(Rsa::generate(2048).map_err(other_error)?).map_err(other_error)?;
    let leaf_key =
        PKey::from_rsa(Rsa::generate(2048).map_err(other_error)?).map_err(other_error)?;

    let mut root_name = X509NameBuilder::new().map_err(other_error)?;
    root_name
        .append_entry_by_text("CN", "Collaborative Benchmark Root")
        .map_err(other_error)?;
    let root_name = root_name.build();

    let mut root_builder = X509::builder().map_err(other_error)?;
    root_builder.set_version(2).map_err(other_error)?;
    let root_serial = BigNum::from_u32(1)
        .and_then(|bn| bn.to_asn1_integer())
        .map_err(other_error)?;
    root_builder
        .set_serial_number(&root_serial)
        .map_err(other_error)?;
    root_builder
        .set_subject_name(&root_name)
        .map_err(other_error)?;
    root_builder
        .set_issuer_name(&root_name)
        .map_err(other_error)?;
    root_builder.set_pubkey(&root_key).map_err(other_error)?;
    let root_not_before = Asn1Time::days_from_now(0).map_err(other_error)?;
    let root_not_after = Asn1Time::days_from_now(365).map_err(other_error)?;
    root_builder
        .set_not_before(&root_not_before)
        .map_err(other_error)?;
    root_builder
        .set_not_after(&root_not_after)
        .map_err(other_error)?;
    root_builder
        .append_extension(
            BasicConstraints::new()
                .critical()
                .ca()
                .build()
                .map_err(other_error)?,
        )
        .map_err(other_error)?;
    root_builder
        .append_extension(
            KeyUsage::new()
                .critical()
                .key_cert_sign()
                .crl_sign()
                .build()
                .map_err(other_error)?,
        )
        .map_err(other_error)?;
    root_builder
        .sign(&root_key, MessageDigest::sha256())
        .map_err(other_error)?;
    let root_cert = root_builder.build();

    Ok(TestIssuer {
        root_pem: pem_string(&root_cert)?,
        root_key,
        root_cert,
        leaf_key,
    })
}

fn issue_test_collaborative_certificate(
    profile: &AggregateCertificateProfile,
    issuer: &TestIssuer,
) -> Result<IssuedAggregateCertificate> {
    let mut leaf_name = X509NameBuilder::new().map_err(other_error)?;
    leaf_name
        .append_entry_by_text("CN", "Collaborative Benchmark Leaf")
        .map_err(other_error)?;
    let leaf_name = leaf_name.build();

    let mut leaf_builder = X509::builder().map_err(other_error)?;
    leaf_builder.set_version(2).map_err(other_error)?;
    let leaf_serial = BigNum::from_u32(2)
        .and_then(|bn| bn.to_asn1_integer())
        .map_err(other_error)?;
    leaf_builder
        .set_serial_number(&leaf_serial)
        .map_err(other_error)?;
    leaf_builder
        .set_subject_name(&leaf_name)
        .map_err(other_error)?;
    leaf_builder
        .set_issuer_name(issuer.root_cert.subject_name())
        .map_err(other_error)?;
    leaf_builder
        .set_pubkey(&issuer.leaf_key)
        .map_err(other_error)?;
    let leaf_not_before = Asn1Time::days_from_now(0).map_err(other_error)?;
    let leaf_not_after = Asn1Time::days_from_now(365).map_err(other_error)?;
    leaf_builder
        .set_not_before(&leaf_not_before)
        .map_err(other_error)?;
    leaf_builder
        .set_not_after(&leaf_not_after)
        .map_err(other_error)?;

    append_utf8_extension(
        &mut leaf_builder,
        ALLOWED_CUSTOM_EXTENSION_OIDS[0],
        &profile.participants_roster_hash,
    )?;
    append_utf8_extension(
        &mut leaf_builder,
        ALLOWED_CUSTOM_EXTENSION_OIDS[1],
        &serde_json::to_string(&profile.participants_references)?,
    )?;
    append_utf8_extension(
        &mut leaf_builder,
        ALLOWED_CUSTOM_EXTENSION_OIDS[2],
        &profile.aggregate_public_key.algorithm,
    )?;
    append_utf8_extension(
        &mut leaf_builder,
        ALLOWED_CUSTOM_EXTENSION_OIDS[3],
        &hex::encode(&profile.aggregate_public_key.bytes),
    )?;
    leaf_builder
        .append_extension(BasicConstraints::new().build().map_err(other_error)?)
        .map_err(other_error)?;
    leaf_builder
        .append_extension(
            KeyUsage::new()
                .digital_signature()
                .build()
                .map_err(other_error)?,
        )
        .map_err(other_error)?;
    leaf_builder
        .sign(&issuer.root_key, MessageDigest::sha256())
        .map_err(other_error)?;
    let leaf_cert = leaf_builder.build();

    Ok(IssuedAggregateCertificate {
        leaf_certificate_pem: pem_string(&leaf_cert)?,
        issuer_certificate_pem: issuer.root_pem.clone(),
    })
}

fn append_utf8_extension(
    builder: &mut openssl::x509::X509Builder,
    oid: &str,
    value: &str,
) -> Result<()> {
    let object = Asn1Object::from_str(oid).map_err(other_error)?;
    let octets = Asn1OctetString::new_from_bytes(&der_utf8_string(value)).map_err(other_error)?;
    let extension = X509Extension::new_from_der(&object, false, &octets).map_err(other_error)?;
    builder.append_extension(extension).map_err(other_error)?;
    Ok(())
}

fn der_utf8_string(text: &str) -> Vec<u8> {
    let mut encoded = Vec::new();
    encoded.push(0x0c);
    let bytes = text.as_bytes();
    if bytes.len() < 0x80 {
        encoded.push(bytes.len() as u8);
    } else {
        let mut len = bytes.len();
        let mut len_bytes = Vec::new();
        while len > 0 {
            len_bytes.push((len & 0xff) as u8);
            len >>= 8;
        }
        len_bytes.reverse();
        encoded.push(0x80 | len_bytes.len() as u8);
        encoded.extend_from_slice(&len_bytes);
    }
    encoded.extend_from_slice(bytes);
    encoded
}

fn pem_string(cert: &X509) -> Result<String> {
    String::from_utf8(cert.to_pem().map_err(other_error)?)
        .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))
}

fn write_samples_csv(mut writer: impl Write, samples: &[Sample]) -> std::io::Result<()> {
    writeln!(
        writer,
        "scenario,participants,run,enroll_ms,finalize_ms,verify_ms,file_sign_ms,file_verify_ms,standard_sign_ms,standard_read_ms,total_ms,output_bytes"
    )?;
    for sample in samples {
        writeln!(
            writer,
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
            sample.scenario,
            sample.participants,
            sample.run,
            sample.enroll_ms,
            sample.finalize_ms,
            sample.verify_ms,
            sample.file_sign_ms,
            sample.file_verify_ms,
            sample.standard_sign_ms,
            sample.standard_read_ms,
            sample.total_ms,
            sample.output_bytes
        )?;
    }
    Ok(())
}

fn print_summary(samples: &[Sample]) {
    println!("summary,scenario,participants,metric,mean_ms,std_ms");
    summarize(samples, "standard_c2pa", 0, "standard_sign_ms", |s| {
        s.standard_sign_ms
    });
    summarize(samples, "standard_c2pa", 0, "standard_read_ms", |s| {
        s.standard_read_ms
    });
    summarize(samples, "standard_c2pa", 0, "total_ms", |s| s.total_ms);

    for &participants in PARTICIPANT_COUNTS {
        let protocol_metrics: [(&str, fn(&Sample) -> f64); 4] = [
            ("enroll_ms", |s: &Sample| s.enroll_ms),
            ("finalize_ms", |s: &Sample| s.finalize_ms),
            ("verify_ms", |s: &Sample| s.verify_ms),
            ("total_ms", |s: &Sample| s.total_ms),
        ];
        for metric in protocol_metrics {
            summarize(samples, "protocol", participants, metric.0, metric.1);
        }
        let file_chain_metrics: [(&str, fn(&Sample) -> f64); 4] = [
            ("enroll_ms", |s: &Sample| s.enroll_ms),
            ("file_sign_ms", |s: &Sample| s.file_sign_ms),
            ("file_verify_ms", |s: &Sample| s.file_verify_ms),
            ("total_ms", |s: &Sample| s.total_ms),
        ];
        for metric in file_chain_metrics {
            summarize(samples, "file_chain", participants, metric.0, metric.1);
        }
    }
}

fn summarize(
    samples: &[Sample],
    scenario: &'static str,
    participants: usize,
    metric: &'static str,
    value: fn(&Sample) -> f64,
) {
    let values = samples
        .iter()
        .filter(|sample| sample.scenario == scenario && sample.participants == participants)
        .map(value)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return;
    }
    let (mean, std_dev) = mean_std(&values);
    println!("{scenario},{participants},{metric},{mean:.4},{std_dev:.4}");
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, variance.sqrt())
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn io_error(kind: std::io::ErrorKind, message: impl Into<String>) -> Error {
    Error::OtherError(Box::new(std::io::Error::new(kind, message.into())))
}

fn other_error<E>(err: E) -> Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    Error::OtherError(Box::new(err))
}
