// Copyright 2026 Adobe. All rights reserved.
// This file is licensed to you under the Apache License,
// Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org/licenses/MIT),
// at your option.

//! Collaborative provenance helpers layered on top of the existing C2PA API.
//!
//! The goal of this module is to keep the current C2PA manifest flow intact
//! while providing reusable data types and validation helpers for a
//! jointly-authorized final state.

use std::{collections::BTreeSet, fs, path::Path};

use openssl::{
    asn1::Asn1Time,
    pkey::Id,
    x509::{store::X509StoreBuilder, verify::X509VerifyFlags, X509StoreContext, X509},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use x509_parser::{extensions::ParsedExtension, pem::parse_x509_pem, prelude::*};

use crate::{Error, Result};

/// Prototype OID arc used by the collaborative extension profile.
pub const CUSTOM_EXTENSION_OID_PREFIX: &str = "1.3.6.1.4.1.55555.1.";

/// Default collaborative extension OIDs used by the current prototype.
pub const ALLOWED_CUSTOM_EXTENSION_OIDS: [&str; 4] = [
    "1.3.6.1.4.1.55555.1.1",
    "1.3.6.1.4.1.55555.1.2",
    "1.3.6.1.4.1.55555.1.3",
    "1.3.6.1.4.1.55555.1.4",
];

/// Manifest assertion label used to carry the collaborative authorization payload.
pub const COLLABORATIVE_AUTHORIZATION_LABEL: &str = "org.contentauth.collaborative.authorization";

/// A participant in the final approver set.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Participant {
    pub identifier: String,
    pub certificate_fingerprint: String,
    pub public_key_der: Vec<u8>,
}

/// A proof-of-possession over a participant enrollment challenge.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ProofOfPossession {
    pub participant_id: String,
    pub challenge: Vec<u8>,
    pub signature: Vec<u8>,
}

/// A signed enrollment package.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct EnrollmentPackage {
    pub participant: Participant,
    pub proof_of_possession: ProofOfPossession,
}

/// A claim digest for the finalized C2PA state.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ClaimDigest(pub Vec<u8>);

/// A partial signature contributed by one participant.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct PartialSignature {
    pub participant_id: String,
    pub bytes: Vec<u8>,
}

/// A collective signature over one claim digest.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct AggregateSignature {
    pub algorithm: String,
    pub bytes: Vec<u8>,
}

/// An aggregated verification key.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct AggregatePublicKey {
    pub algorithm: String,
    pub bytes: Vec<u8>,
}

/// The certificate-layer profile bound to one final approver set.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct AggregateCertificateProfile {
    pub aggregate_public_key: AggregatePublicKey,
    pub participants_roster_hash: String,
    pub participants_references: Vec<String>,
}

/// A locally issued aggregate-certificate artifact.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct IssuedAggregateCertificate {
    pub leaf_certificate_pem: String,
    pub issuer_certificate_pem: String,
}

/// The embedded authorization object stored alongside the signed asset.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct AuthorizationPackage {
    pub claim_digest: ClaimDigest,
    pub aggregate_signature: AggregateSignature,
    pub aggregate_certificate_hash: String,
}

/// The full collaborative payload embedded into the signed asset.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct EmbeddedCollaborativeManifest {
    pub authorization: AuthorizationPackage,
    pub aggregate_certificate: IssuedAggregateCertificate,
}

/// Summary returned by collaborative verification.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct VerificationSummary {
    pub verified: bool,
    pub approver_ids: Vec<String>,
    pub aggregate_algorithm: String,
}

/// Certificate verification route selected from the certificate's key fields.
///
/// The standard C2PA path remains unchanged. Collaborative certificates are
/// identified by the prototype extension OID arc and then processed by the
/// collaborative branch.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
pub enum CertificateVerificationRoute {
    StandardC2pa,
    Collaborative,
}

/// Minimal collective-signature backend interface used by the collaborative layer.
///
/// The crate does not impose a specific multisignature backend. Implementors can
/// use BLS, a testing backend, or another collective-signature scheme as long as
/// it is consistent with the roster-binding rules enforced here.
pub trait CollectiveSigner {
    fn algorithm(&self) -> &'static str;

    fn aggregate_public_key(&self, roster: &[Participant]) -> Result<AggregatePublicKey, Error>;

    fn partial_sign(
        &self,
        participant: &Participant,
        claim_digest: &ClaimDigest,
    ) -> Result<PartialSignature, Error>;

    fn aggregate_signatures(
        &self,
        claim_digest: &ClaimDigest,
        roster: &[Participant],
        partials: &[PartialSignature],
    ) -> Result<AggregateSignature, Error>;

    fn verify(
        &self,
        claim_digest: &ClaimDigest,
        aggregate_public_key: &AggregatePublicKey,
        signature: &AggregateSignature,
        roster: &[Participant],
    ) -> Result<(), Error>;
}

/// Canonicalize a roster by sorting it and rejecting duplicate identities or fingerprints.
pub fn canonicalize_roster(roster: &[Participant]) -> Result<Vec<Participant>, Error> {
    if roster.is_empty() {
        return Err(Error::OtherError(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "approver roster is empty",
        ))));
    }

    let mut canonical = roster.to_vec();
    canonical.sort_by(|left, right| {
        left.identifier.cmp(&right.identifier).then(
            left.certificate_fingerprint
                .cmp(&right.certificate_fingerprint),
        )
    });

    for pair in canonical.windows(2) {
        if pair[0].identifier == pair[1].identifier {
            return Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("duplicate participant in roster: {}", pair[0].identifier),
            ))));
        }
        if pair[0].certificate_fingerprint == pair[1].certificate_fingerprint {
            return Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "duplicate participant fingerprint in roster: {}",
                    pair[0].certificate_fingerprint
                ),
            ))));
        }
    }

    Ok(canonical)
}

/// Encode a participant deterministically for hashing and equality checks.
pub fn encode_participant(participant: &Participant) -> Vec<u8> {
    let mut encoded = Vec::new();
    append_len_prefixed(&mut encoded, participant.identifier.as_bytes());
    append_len_prefixed(&mut encoded, participant.certificate_fingerprint.as_bytes());
    append_len_prefixed(&mut encoded, &participant.public_key_der);
    encoded
}

/// Encode a roster deterministically.
pub fn encode_roster(roster: &[Participant]) -> Result<Vec<u8>, Error> {
    let canonical = canonicalize_roster(roster)?;
    let mut encoded = Vec::new();
    for participant in canonical {
        append_len_prefixed(&mut encoded, &encode_participant(&participant));
    }
    Ok(encoded)
}

/// Compute the roster commitment used to bind a certificate to one final approver set.
pub fn roster_hash_hex(roster: &[Participant]) -> Result<String, Error> {
    let roster_bytes = encode_roster(roster)?;
    Ok(hex::encode(Sha256::digest(&roster_bytes)))
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

/// Build a certificate profile from a roster and aggregated public key.
pub fn build_aggregate_certificate_profile(
    aggregate_public_key: AggregatePublicKey,
    roster: &[Participant],
) -> Result<AggregateCertificateProfile, Error> {
    let canonical = canonicalize_roster(roster)?;
    let participants_references = canonical
        .iter()
        .map(|participant| participant.certificate_fingerprint.clone())
        .collect();

    Ok(AggregateCertificateProfile {
        aggregate_public_key,
        participants_roster_hash: roster_hash_hex(roster)?,
        participants_references,
    })
}

/// Validate that a certificate profile matches the signer and roster.
pub fn validate_certificate_profile<S: CollectiveSigner>(
    signer: &S,
    profile: &AggregateCertificateProfile,
    roster: &[Participant],
) -> Result<(), Error> {
    let canonical = canonicalize_roster(roster)?;
    let expected_apk = signer.aggregate_public_key(roster)?;
    let expected_hash = roster_hash_hex(roster)?;
    let expected_references = canonical
        .iter()
        .map(|participant| participant.certificate_fingerprint.clone())
        .collect::<Vec<_>>();

    if profile.aggregate_public_key == expected_apk
        && profile.participants_roster_hash == expected_hash
        && profile.participants_references == expected_references
    {
        Ok(())
    } else {
        Err(Error::OtherError(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "certificate profile mismatch",
        ))))
    }
}

/// A simple trust store for collaborative certificate validation.
#[derive(Debug, Clone, Default)]
pub struct TrustStore {
    anchors_pem: Vec<String>,
}

impl TrustStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_pem(anchor_pem: impl Into<String>) -> Result<Self, Error> {
        Self::from_pems([anchor_pem])
    }

    pub fn from_pems<I, S>(anchors: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Ok(Self {
            anchors_pem: normalize_trust_anchors(
                anchors.into_iter().map(Into::into).collect::<Vec<_>>(),
            )?,
        })
    }

    pub fn from_pem_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let anchor_pem = fs::read_to_string(path.as_ref())?;
        Ok(Self {
            anchors_pem: normalize_trust_anchors(vec![anchor_pem])?,
        })
    }

    pub fn from_pem_directory(path: impl AsRef<Path>) -> Result<Self, Error> {
        let mut entries =
            fs::read_dir(path.as_ref())?.collect::<std::result::Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.path());

        let mut anchors_pem = Vec::new();
        for entry in entries {
            let file_type = entry.file_type()?;
            if !file_type.is_file() {
                continue;
            }
            anchors_pem.push(fs::read_to_string(entry.path())?);
        }

        Ok(Self {
            anchors_pem: normalize_trust_anchors(anchors_pem)?,
        })
    }

    pub fn add_pem(&mut self, anchor_pem: impl Into<String>) -> Result<(), Error> {
        let mut anchors = self.anchors_pem.clone();
        anchors.push(anchor_pem.into());
        self.anchors_pem = normalize_trust_anchors(anchors)?;
        Ok(())
    }

    pub fn anchors(&self) -> &[String] {
        &self.anchors_pem
    }

    pub fn ensure_non_empty(&self) -> Result<(), Error> {
        if self.anchors_pem.is_empty() {
            Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "no trust anchors configured",
            ))))
        } else {
            Ok(())
        }
    }
}

/// Verifier policy for collaborative certificate and signature checks.
#[derive(Debug, Clone)]
pub struct VerifierPolicy {
    allowed_aggregate_algorithms: BTreeSet<String>,
    allowed_custom_extension_oids: BTreeSet<String>,
    allowed_leaf_key_types: BTreeSet<i32>,
    minimum_leaf_key_bits: u32,
    maximum_certificate_validity_days: i32,
    check_current_time: bool,
}

impl VerifierPolicy {
    pub fn strict_for_algorithm(algorithm: impl Into<String>) -> Self {
        Self {
            allowed_aggregate_algorithms: BTreeSet::from([algorithm.into()]),
            allowed_custom_extension_oids: ALLOWED_CUSTOM_EXTENSION_OIDS
                .iter()
                .map(|oid| oid.to_string())
                .collect(),
            allowed_leaf_key_types: BTreeSet::from([Id::RSA.as_raw()]),
            minimum_leaf_key_bits: 2048,
            maximum_certificate_validity_days: 398,
            check_current_time: true,
        }
    }

    pub fn allow_aggregate_algorithm(&mut self, algorithm: impl Into<String>) {
        self.allowed_aggregate_algorithms.insert(algorithm.into());
    }

    pub fn set_allowed_aggregate_algorithms<I, S>(&mut self, algorithms: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_aggregate_algorithms = algorithms.into_iter().map(Into::into).collect();
    }

    pub fn allow_custom_extension_oid(&mut self, oid: impl Into<String>) {
        self.allowed_custom_extension_oids.insert(oid.into());
    }

    pub fn set_allowed_custom_extension_oids<I, S>(&mut self, oids: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_custom_extension_oids = oids.into_iter().map(Into::into).collect();
    }

    pub fn set_check_current_time(&mut self, check_current_time: bool) {
        self.check_current_time = check_current_time;
    }

    pub fn allow_leaf_key_type(&mut self, key_type: Id) {
        self.allowed_leaf_key_types.insert(key_type.as_raw());
    }

    pub fn set_allowed_leaf_key_types<I>(&mut self, key_types: I)
    where
        I: IntoIterator<Item = Id>,
    {
        self.allowed_leaf_key_types = key_types.into_iter().map(|id| id.as_raw()).collect();
    }

    pub fn set_minimum_leaf_key_bits(&mut self, minimum_leaf_key_bits: u32) {
        self.minimum_leaf_key_bits = minimum_leaf_key_bits;
    }

    pub fn set_maximum_certificate_validity_days(
        &mut self,
        maximum_certificate_validity_days: i32,
    ) {
        self.maximum_certificate_validity_days = maximum_certificate_validity_days;
    }

    fn validate_algorithms(
        &self,
        signature_algorithm: &str,
        public_key_algorithm: &str,
    ) -> Result<(), Error> {
        if !self
            .allowed_aggregate_algorithms
            .contains(signature_algorithm)
        {
            return Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "aggregate algorithm is not allowed by verifier policy: {signature_algorithm}"
                ),
            ))));
        }
        if !self
            .allowed_aggregate_algorithms
            .contains(public_key_algorithm)
        {
            return Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "aggregate algorithm is not allowed by verifier policy: {public_key_algorithm}"
                ),
            ))));
        }
        Ok(())
    }

    fn validate_custom_extension_oids(&self, extension_oids: &[String]) -> Result<(), Error> {
        for oid in extension_oids {
            if oid.starts_with(CUSTOM_EXTENSION_OID_PREFIX)
                && !self.allowed_custom_extension_oids.contains(oid)
            {
                return Err(Error::OtherError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("certificate contains a disallowed custom extension oid: {oid}"),
                ))));
            }
        }
        Ok(())
    }

    fn validate_leaf_certificate(&self, certificate: &X509) -> Result<(), Error> {
        let public_key = certificate.public_key().map_err(other_error)?;
        let key_type = public_key.id().as_raw();
        if !self.allowed_leaf_key_types.contains(&key_type) {
            return Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("disallowed leaf key type {key_type}"),
            ))));
        }
        if public_key.bits() < self.minimum_leaf_key_bits {
            return Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("leaf key too small: {} bits", public_key.bits()),
            ))));
        }

        let not_before = certificate.not_before();
        let not_after = certificate.not_after();
        if not_after < not_before {
            return Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "certificate validity interval is inverted",
            ))));
        }

        let validity = not_before.diff(not_after).map_err(other_error)?;
        if validity.days > self.maximum_certificate_validity_days {
            return Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "certificate validity exceeds policy: {} days",
                    validity.days
                ),
            ))));
        }

        if self.check_current_time {
            let now = Asn1Time::days_from_now(0).map_err(other_error)?;
            if not_before > &now || not_after < &now {
                return Err(Error::OtherError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "certificate is not valid at the current time",
                ))));
            }
        }

        Ok(())
    }
}

/// Verifier for the collaborative layer.
pub struct CollaborativeVerifier<S> {
    signer: S,
    trust_store: TrustStore,
    policy: VerifierPolicy,
}

impl<S> CollaborativeVerifier<S>
where
    S: CollectiveSigner,
{
    pub fn new(signer: S, trust_anchor_pem: impl Into<String>) -> Result<Self, Error> {
        Ok(Self::with_trust_store(
            signer,
            TrustStore::from_pem(trust_anchor_pem)?,
        ))
    }

    pub fn with_trust_store(signer: S, trust_store: TrustStore) -> Self {
        let policy = VerifierPolicy::strict_for_algorithm(signer.algorithm());
        Self::with_trust_store_and_policy(signer, trust_store, policy)
    }

    pub fn with_trust_store_and_policy(
        signer: S,
        trust_store: TrustStore,
        policy: VerifierPolicy,
    ) -> Self {
        Self {
            signer,
            trust_store,
            policy,
        }
    }

    pub fn verify_embedded(
        &self,
        manifest: &EmbeddedCollaborativeManifest,
        roster: &[Participant],
    ) -> Result<VerificationSummary, Error> {
        let expected_certificate_hash = hex::encode(Sha256::digest(serde_json::to_vec(
            &manifest.aggregate_certificate,
        )?));
        if expected_certificate_hash != manifest.authorization.aggregate_certificate_hash {
            return Err(Error::OtherError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "certificate hash mismatch",
            ))));
        }

        self.trust_store.ensure_non_empty()?;
        let aggregate_certificate = validate_issued_certificate(
            &self.signer,
            &manifest.aggregate_certificate,
            roster,
            self.trust_store.anchors(),
            self.policy.check_current_time,
        )?;
        let extension_oids = extract_extension_oids(&manifest.aggregate_certificate)?;
        self.policy
            .validate_custom_extension_oids(&extension_oids)?;
        let leaf_certificate = parse_leaf_certificate(&manifest.aggregate_certificate)?;
        self.policy.validate_leaf_certificate(&leaf_certificate)?;
        self.verify_with_certificate_profile(
            &manifest.authorization,
            roster,
            &aggregate_certificate,
        )
    }

    pub fn verify_with_certificate_profile(
        &self,
        package: &AuthorizationPackage,
        roster: &[Participant],
        aggregate_certificate: &AggregateCertificateProfile,
    ) -> Result<VerificationSummary, Error> {
        self.policy.validate_algorithms(
            &package.aggregate_signature.algorithm,
            &aggregate_certificate.aggregate_public_key.algorithm,
        )?;
        validate_certificate_profile(&self.signer, aggregate_certificate, roster)?;
        self.signer.verify(
            &package.claim_digest,
            &aggregate_certificate.aggregate_public_key,
            &package.aggregate_signature,
            roster,
        )?;

        Ok(VerificationSummary {
            verified: true,
            approver_ids: canonicalize_roster(roster)?
                .iter()
                .map(|participant| participant.identifier.clone())
                .collect(),
            aggregate_algorithm: package.aggregate_signature.algorithm.clone(),
        })
    }
}

/// Validate the aggregate certificate and return the extracted profile.
pub fn validate_issued_certificate<S: CollectiveSigner>(
    signer: &S,
    issued: &IssuedAggregateCertificate,
    roster: &[Participant],
    trust_anchors_pem: &[String],
    check_current_time: bool,
) -> Result<AggregateCertificateProfile, Error> {
    verify_certificate_chain(issued, trust_anchors_pem, check_current_time)?;
    let profile = extract_certificate_profile(issued)?;
    validate_certificate_profile(signer, &profile, roster)?;
    Ok(profile)
}

fn parse_leaf_certificate(issued: &IssuedAggregateCertificate) -> Result<X509, Error> {
    X509::from_pem(issued.leaf_certificate_pem.as_bytes()).map_err(other_error)
}

/// Verify the issuer chain against the supplied trust anchors.
pub fn verify_certificate_chain(
    issued: &IssuedAggregateCertificate,
    trust_anchors_pem: &[String],
    check_current_time: bool,
) -> Result<(), Error> {
    let leaf = X509::from_pem(issued.leaf_certificate_pem.as_bytes()).map_err(other_error)?;
    let embedded_issuer =
        X509::from_pem(issued.issuer_certificate_pem.as_bytes()).map_err(other_error)?;
    let embedded_issuer_der = embedded_issuer.to_der().map_err(other_error)?;

    for trust_anchor_pem in trust_anchors_pem {
        let trust_anchor = X509::from_pem(trust_anchor_pem.as_bytes()).map_err(other_error)?;
        let trust_anchor_der = trust_anchor.to_der().map_err(other_error)?;
        if embedded_issuer_der != trust_anchor_der {
            continue;
        }

        let mut store_builder = X509StoreBuilder::new().map_err(other_error)?;
        if !check_current_time {
            store_builder
                .set_flags(X509VerifyFlags::NO_CHECK_TIME)
                .map_err(other_error)?;
        }
        store_builder.add_cert(trust_anchor).map_err(other_error)?;
        let store = store_builder.build();
        let chain = openssl::stack::Stack::<X509>::new().map_err(other_error)?;
        let mut context = X509StoreContext::new().map_err(other_error)?;
        let verified = context
            .init(&store, &leaf, &chain, |context| context.verify_cert())
            .map_err(other_error)?;

        if verified {
            return Ok(());
        }
    }

    Err(Error::OtherError(Box::new(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        "certificate issuer does not match the configured trust anchor",
    ))))
}

/// Extract the collaborative certificate profile from an issued aggregate certificate.
pub fn extract_certificate_profile(
    issued: &IssuedAggregateCertificate,
) -> Result<AggregateCertificateProfile, Error> {
    let (_, pem) = parse_x509_pem(issued.leaf_certificate_pem.as_bytes())
        .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))?;
    let certificate = pem
        .parse_x509()
        .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))?;

    let roster_hash = find_utf8_extension(&certificate, ALLOWED_CUSTOM_EXTENSION_OIDS[0])?;
    let participant_refs_json =
        find_utf8_extension(&certificate, ALLOWED_CUSTOM_EXTENSION_OIDS[1])?;
    let aggregate_algorithm = find_utf8_extension(&certificate, ALLOWED_CUSTOM_EXTENSION_OIDS[2])?;
    let aggregate_public_key_hex =
        find_utf8_extension(&certificate, ALLOWED_CUSTOM_EXTENSION_OIDS[3])?;

    Ok(AggregateCertificateProfile {
        aggregate_public_key: AggregatePublicKey {
            algorithm: aggregate_algorithm,
            bytes: hex::decode(aggregate_public_key_hex)
                .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))?,
        },
        participants_roster_hash: roster_hash,
        participants_references: serde_json::from_str(&participant_refs_json)?,
    })
}

/// Return the set of extension OIDs present in the certificate.
pub fn extract_extension_oids(issued: &IssuedAggregateCertificate) -> Result<Vec<String>, Error> {
    let (_, pem) = parse_x509_pem(issued.leaf_certificate_pem.as_bytes())
        .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))?;
    let certificate = pem
        .parse_x509()
        .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))?;

    Ok(certificate
        .tbs_certificate
        .extensions()
        .iter()
        .map(|extension| extension.oid.to_id_string())
        .collect())
}

/// Decide which verification chain should be used for an issued certificate.
///
/// This is a pure routing helper: it does not alter the existing C2PA
/// certificate validation flow. Callers can inspect the route and then choose
/// between the standard trust policy path and the collaborative path.
pub fn certificate_verification_route(
    issued: &IssuedAggregateCertificate,
) -> Result<CertificateVerificationRoute, Error> {
    let extension_oids = extract_extension_oids(issued)?;
    if extension_oids
        .iter()
        .any(|oid| oid.starts_with(CUSTOM_EXTENSION_OID_PREFIX))
    {
        Ok(CertificateVerificationRoute::Collaborative)
    } else {
        Ok(CertificateVerificationRoute::StandardC2pa)
    }
}

/// Verify an issued certificate by routing on its key fields.
///
/// Standard certificates are verified with the existing chain-validation logic.
/// Collaborative certificates are verified with the collaborative path without
/// changing the original C2PA certificate verification implementation.
pub fn verify_issued_certificate_by_route<S: CollectiveSigner>(
    signer: &S,
    issued: &IssuedAggregateCertificate,
    roster: &[Participant],
    trust_anchors_pem: &[String],
    check_current_time: bool,
) -> Result<CertificateVerificationRoute, Error> {
    match certificate_verification_route(issued)? {
        CertificateVerificationRoute::StandardC2pa => {
            verify_certificate_chain(issued, trust_anchors_pem, check_current_time)?;
            Ok(CertificateVerificationRoute::StandardC2pa)
        }
        CertificateVerificationRoute::Collaborative => {
            validate_issued_certificate(
                signer,
                issued,
                roster,
                trust_anchors_pem,
                check_current_time,
            )?;
            Ok(CertificateVerificationRoute::Collaborative)
        }
    }
}

fn normalize_trust_anchors(raw_pems: Vec<String>) -> Result<Vec<String>, Error> {
    let mut dedup = BTreeSet::new();
    let mut normalized = Vec::new();

    for raw_pem in raw_pems {
        let certificates = X509::stack_from_pem(raw_pem.as_bytes()).map_err(other_error)?;
        if certificates.is_empty() {
            return Err(io_error(
                std::io::ErrorKind::InvalidInput,
                "pem payload does not contain any certificate",
            ));
        }

        for certificate in certificates {
            validate_trust_anchor_certificate(&certificate)?;
            let der = certificate.to_der().map_err(other_error)?;
            if dedup.insert(der.clone()) {
                normalized.push(
                    String::from_utf8(certificate.to_pem().map_err(other_error)?).map_err(
                        |err| io_error(std::io::ErrorKind::InvalidData, err.to_string()),
                    )?,
                );
            }
        }
    }

    Ok(normalized)
}

fn validate_trust_anchor_certificate(certificate: &X509) -> Result<(), Error> {
    let der = certificate.to_der().map_err(other_error)?;
    let (_, parsed) = parse_x509_certificate(&der)
        .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))?;

    if parsed.tbs_certificate.subject.as_raw() != parsed.tbs_certificate.issuer.as_raw() {
        return Err(io_error(
            std::io::ErrorKind::InvalidInput,
            "trust anchor must be self-issued",
        ));
    }

    let mut saw_basic_constraints = false;
    let mut saw_key_usage = false;
    for extension in parsed.tbs_certificate.extensions() {
        match extension.parsed_extension() {
            ParsedExtension::BasicConstraints(basic_constraints) => {
                saw_basic_constraints = true;
                if !basic_constraints.ca {
                    return Err(io_error(
                        std::io::ErrorKind::InvalidInput,
                        "trust anchor must be a CA certificate",
                    ));
                }
            }
            ParsedExtension::KeyUsage(key_usage) => {
                saw_key_usage = true;
                if !key_usage.key_cert_sign() {
                    return Err(io_error(
                        std::io::ErrorKind::InvalidInput,
                        "trust anchor must allow certificate signing",
                    ));
                }
            }
            ParsedExtension::ParseError { error } => {
                return Err(io_error(
                    std::io::ErrorKind::InvalidData,
                    format!("failed to parse trust anchor extension: {error}"),
                ));
            }
            _ => {}
        }
    }

    if !saw_basic_constraints {
        return Err(io_error(
            std::io::ErrorKind::InvalidInput,
            "trust anchor missing basic constraints",
        ));
    }
    if !saw_key_usage {
        return Err(io_error(
            std::io::ErrorKind::InvalidInput,
            "trust anchor missing key usage",
        ));
    }

    Ok(())
}

fn find_utf8_extension(certificate: &X509Certificate<'_>, oid: &str) -> Result<String, Error> {
    let extension = certificate
        .tbs_certificate
        .extensions()
        .iter()
        .find(|extension| extension.oid.to_id_string() == oid)
        .ok_or_else(|| {
            io_error(
                std::io::ErrorKind::InvalidData,
                format!("missing certificate extension {oid}"),
            )
        })?;

    parse_der_utf8_string(extension.value)
}

fn parse_der_utf8_string(input: &[u8]) -> Result<String, Error> {
    if input.first().copied() != Some(0x0c) {
        return Err(io_error(
            std::io::ErrorKind::InvalidData,
            "expected DER UTF8String extension value",
        ));
    }

    let (length, header_len) = parse_der_length(
        input
            .get(1..)
            .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "invalid DER length"))?,
    )?;
    let start = 1 + header_len;
    let end = start + length;
    let content = input
        .get(start..end)
        .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "truncated DER UTF8String"))?;
    String::from_utf8(content.to_vec())
        .map_err(|err| io_error(std::io::ErrorKind::InvalidData, err.to_string()))
}

fn parse_der_length(input: &[u8]) -> Result<(usize, usize), Error> {
    let first = *input
        .first()
        .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "missing DER length"))?;
    if first & 0x80 == 0 {
        Ok((first as usize, 1))
    } else {
        let len_bytes = (first & 0x7f) as usize;
        let bytes = input
            .get(1..1 + len_bytes)
            .ok_or_else(|| io_error(std::io::ErrorKind::InvalidData, "truncated DER length"))?;
        let mut length = 0usize;
        for byte in bytes {
            length = (length << 8) | (*byte as usize);
        }
        Ok((length, 1 + len_bytes))
    }
}

fn append_len_prefixed(buffer: &mut Vec<u8>, bytes: &[u8]) {
    let len = bytes.len() as u32;
    buffer.extend_from_slice(&len.to_be_bytes());
    buffer.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use openssl::{
        asn1::{Asn1Object, Asn1OctetString, Asn1Time},
        bn::BigNum,
        hash::MessageDigest,
        pkey::PKey,
        rsa::Rsa,
        x509::{
            extension::{BasicConstraints, KeyUsage},
            X509Extension, X509NameBuilder, X509,
        },
    };
    use serde_json::json;

    use crate::{
        utils::{io_utils::patch_stream, test::write_jpeg_placeholder_stream, test_signer::test_signer},
        Builder, Context, Reader,
    };

    #[derive(Clone, Copy)]
    struct DummySigner;

    impl CollectiveSigner for DummySigner {
        fn algorithm(&self) -> &'static str {
            "demo-bdn-placeholder"
        }

        fn aggregate_public_key(
            &self,
            roster: &[Participant],
        ) -> Result<AggregatePublicKey, Error> {
            Ok(AggregatePublicKey {
                algorithm: self.algorithm().to_owned(),
                bytes: Sha256::digest(encode_roster(roster)?).to_vec(),
            })
        }

        fn partial_sign(
            &self,
            participant: &Participant,
            claim_digest: &ClaimDigest,
        ) -> Result<PartialSignature, Error> {
            Ok(PartialSignature {
                participant_id: participant.identifier.clone(),
                bytes: Sha256::digest(
                    [participant.public_key_der.as_slice(), &claim_digest.0].concat(),
                )
                .to_vec(),
            })
        }

        fn aggregate_signatures(
            &self,
            claim_digest: &ClaimDigest,
            roster: &[Participant],
            partials: &[PartialSignature],
        ) -> Result<AggregateSignature, Error> {
            let canonical = canonicalize_roster(roster)?;
            let mut material = claim_digest.0.clone();
            for participant in canonical {
                let partial = partials
                    .iter()
                    .find(|partial| partial.participant_id == participant.identifier)
                    .ok_or_else(|| {
                        Error::OtherError(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "missing partial signature",
                        )))
                    })?;
                material.extend_from_slice(&partial.bytes);
            }
            Ok(AggregateSignature {
                algorithm: self.algorithm().to_owned(),
                bytes: Sha256::digest(material).to_vec(),
            })
        }

        fn verify(
            &self,
            claim_digest: &ClaimDigest,
            aggregate_public_key: &AggregatePublicKey,
            signature: &AggregateSignature,
            roster: &[Participant],
        ) -> Result<(), Error> {
            let expected_apk = self.aggregate_public_key(roster)?;
            if expected_apk != *aggregate_public_key {
                return Err(Error::OtherError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "aggregate public key mismatch",
                ))));
            }
            let partials = roster
                .iter()
                .map(|participant| self.partial_sign(participant, claim_digest))
                .collect::<Result<Vec<_>, _>>()?;
            let expected_sigma = self.aggregate_signatures(claim_digest, roster, &partials)?;
            if expected_sigma != *signature {
                return Err(Error::OtherError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "aggregate signature verification failed",
                ))));
            }
            Ok(())
        }
    }

    #[test]
    fn canonical_roster_is_ordered_and_unique() {
        let roster = vec![
            Participant {
                identifier: "bob".to_string(),
                certificate_fingerprint: "fp-bob".to_string(),
                public_key_der: vec![2],
            },
            Participant {
                identifier: "alice".to_string(),
                certificate_fingerprint: "fp-alice".to_string(),
                public_key_der: vec![1],
            },
        ];

        let canonical = canonicalize_roster(&roster).unwrap();
        assert_eq!(canonical[0].identifier, "alice");
        assert_eq!(canonical[1].identifier, "bob");
    }

    #[test]
    fn validate_certificate_profile_matches_expected_roster() {
        let signer = DummySigner;
        let roster = vec![
            Participant {
                identifier: "alice".to_string(),
                certificate_fingerprint: "fp-alice".to_string(),
                public_key_der: vec![1],
            },
            Participant {
                identifier: "bob".to_string(),
                certificate_fingerprint: "fp-bob".to_string(),
                public_key_der: vec![2],
            },
        ];

        let profile = build_aggregate_certificate_profile(
            signer.aggregate_public_key(&roster).unwrap(),
            &roster,
        )
        .unwrap();

        assert!(validate_certificate_profile(&signer, &profile, &roster).is_ok());
    }

    #[test]
    fn validate_certificate_profile_rejects_reordered_references() {
        let signer = DummySigner;
        let roster = vec![
            Participant {
                identifier: "alice".to_string(),
                certificate_fingerprint: "fp-alice".to_string(),
                public_key_der: vec![1],
            },
            Participant {
                identifier: "bob".to_string(),
                certificate_fingerprint: "fp-bob".to_string(),
                public_key_der: vec![2],
            },
        ];

        let mut profile = build_aggregate_certificate_profile(
            signer.aggregate_public_key(&roster).unwrap(),
            &roster,
        )
        .unwrap();
        profile.participants_references.swap(0, 1);

        assert!(validate_certificate_profile(&signer, &profile, &roster).is_err());
    }

    #[test]
    fn trust_store_rejects_invalid_pem() {
        assert!(TrustStore::from_pem("not a certificate").is_err());
    }

    fn simple_manifest_json() -> String {
        json!({
            "claim_generator_info": [
                {
                    "name": "c2pa_test",
                    "version": "1.0.0"
                }
            ],
            "title": "Collaborative Test Manifest",
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "digitalSourceType": "http://c2pa.org/digitalsourcetype/empty",
                            }
                        ]
                    }
                }
            ]
        })
        .to_string()
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

    fn issue_test_collaborative_certificate(
        profile: &AggregateCertificateProfile,
    ) -> Result<(IssuedAggregateCertificate, String)> {
        let root_key = PKey::from_rsa(Rsa::generate(2048).map_err(other_error)?)
            .map_err(other_error)?;
        let leaf_key = PKey::from_rsa(Rsa::generate(2048).map_err(other_error)?)
            .map_err(other_error)?;

        let mut root_name = X509NameBuilder::new().map_err(other_error)?;
        root_name
            .append_entry_by_text("CN", "Collaborative Test Root")
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
            .append_extension(BasicConstraints::new().critical().ca().build().map_err(other_error)?)
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

        let mut leaf_name = X509NameBuilder::new().map_err(other_error)?;
        leaf_name
            .append_entry_by_text("CN", "Collaborative Aggregate Leaf")
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
            .set_issuer_name(root_cert.subject_name())
            .map_err(other_error)?;
        leaf_builder.set_pubkey(&leaf_key).map_err(other_error)?;
        let leaf_not_before = Asn1Time::days_from_now(0).map_err(other_error)?;
        let leaf_not_after = Asn1Time::days_from_now(365).map_err(other_error)?;
        leaf_builder
            .set_not_before(&leaf_not_before)
            .map_err(other_error)?;
        leaf_builder
            .set_not_after(&leaf_not_after)
            .map_err(other_error)?;

        // Encode the profile directly into X.509 custom extensions so the verifier
        // exercises the same bridge design the paper describes.
        let roster_hash_oid = Asn1Object::from_str(ALLOWED_CUSTOM_EXTENSION_OIDS[0])
            .map_err(other_error)?;
        let roster_hash_value = Asn1OctetString::new_from_bytes(&der_utf8_string(
            &profile.participants_roster_hash,
        ))
        .map_err(other_error)?;
        let roster_hash_ext = X509Extension::new_from_der(
            &roster_hash_oid,
            false,
            &roster_hash_value,
        )
        .map_err(other_error)?;
        leaf_builder
            .append_extension(roster_hash_ext)
            .map_err(other_error)?;

        let participant_refs_json =
            serde_json::to_string(&profile.participants_references).map_err(other_error)?;
        let participant_refs_oid = Asn1Object::from_str(ALLOWED_CUSTOM_EXTENSION_OIDS[1])
            .map_err(other_error)?;
        let participant_refs_value = Asn1OctetString::new_from_bytes(&der_utf8_string(
            &participant_refs_json,
        ))
        .map_err(other_error)?;
        let participant_refs_ext = X509Extension::new_from_der(
            &participant_refs_oid,
            false,
            &participant_refs_value,
        )
        .map_err(other_error)?;
        leaf_builder
            .append_extension(participant_refs_ext)
            .map_err(other_error)?;

        let aggregate_algorithm_oid = Asn1Object::from_str(ALLOWED_CUSTOM_EXTENSION_OIDS[2])
            .map_err(other_error)?;
        let aggregate_algorithm_value = Asn1OctetString::new_from_bytes(&der_utf8_string(
            &profile.aggregate_public_key.algorithm,
        ))
        .map_err(other_error)?;
        let aggregate_algorithm_ext = X509Extension::new_from_der(
            &aggregate_algorithm_oid,
            false,
            &aggregate_algorithm_value,
        )
        .map_err(other_error)?;
        leaf_builder
            .append_extension(aggregate_algorithm_ext)
            .map_err(other_error)?;

        let aggregate_public_key_hex = hex::encode(&profile.aggregate_public_key.bytes);
        let aggregate_public_key_oid = Asn1Object::from_str(ALLOWED_CUSTOM_EXTENSION_OIDS[3])
            .map_err(other_error)?;
        let aggregate_public_key_value = Asn1OctetString::new_from_bytes(&der_utf8_string(
            &aggregate_public_key_hex,
        ))
        .map_err(other_error)?;
        let aggregate_public_key_ext = X509Extension::new_from_der(
            &aggregate_public_key_oid,
            false,
            &aggregate_public_key_value,
        )
        .map_err(other_error)?;
        leaf_builder
            .append_extension(aggregate_public_key_ext)
            .map_err(other_error)?;

        // Keep a familiar leaf profile: the verifier policy still expects an RSA leaf.
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
            .sign(&root_key, MessageDigest::sha256())
            .map_err(other_error)?;
        let leaf_cert = leaf_builder.build();

        let issued = IssuedAggregateCertificate {
            leaf_certificate_pem: pem_string(&leaf_cert)?,
            issuer_certificate_pem: pem_string(&root_cert)?,
        };

        Ok((issued, pem_string(&root_cert)?))
    }

    fn issue_test_standard_certificate() -> Result<(IssuedAggregateCertificate, String)> {
        let root_key =
            PKey::from_rsa(Rsa::generate(2048).map_err(other_error)?).map_err(other_error)?;
        let leaf_key =
            PKey::from_rsa(Rsa::generate(2048).map_err(other_error)?).map_err(other_error)?;

        let mut root_name = X509NameBuilder::new().map_err(other_error)?;
        root_name
            .append_entry_by_text("CN", "Standard Test Root")
            .map_err(other_error)?;
        let root_name = root_name.build();

        let mut root_builder = X509::builder().map_err(other_error)?;
        root_builder.set_version(2).map_err(other_error)?;
        let root_serial = BigNum::from_u32(11)
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
            .append_extension(BasicConstraints::new().critical().ca().build().map_err(other_error)?)
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

        let mut leaf_name = X509NameBuilder::new().map_err(other_error)?;
        leaf_name
            .append_entry_by_text("CN", "Standard Test Leaf")
            .map_err(other_error)?;
        let leaf_name = leaf_name.build();

        let mut leaf_builder = X509::builder().map_err(other_error)?;
        leaf_builder.set_version(2).map_err(other_error)?;
        let leaf_serial = BigNum::from_u32(12)
            .and_then(|bn| bn.to_asn1_integer())
            .map_err(other_error)?;
        leaf_builder
            .set_serial_number(&leaf_serial)
            .map_err(other_error)?;
        leaf_builder
            .set_subject_name(&leaf_name)
            .map_err(other_error)?;
        leaf_builder
            .set_issuer_name(root_cert.subject_name())
            .map_err(other_error)?;
        leaf_builder.set_pubkey(&leaf_key).map_err(other_error)?;
        let leaf_not_before = Asn1Time::days_from_now(0).map_err(other_error)?;
        let leaf_not_after = Asn1Time::days_from_now(365).map_err(other_error)?;
        leaf_builder
            .set_not_before(&leaf_not_before)
            .map_err(other_error)?;
        leaf_builder
            .set_not_after(&leaf_not_after)
            .map_err(other_error)?;
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
            .sign(&root_key, MessageDigest::sha256())
            .map_err(other_error)?;
        let leaf_cert = leaf_builder.build();

        let issued = IssuedAggregateCertificate {
            leaf_certificate_pem: pem_string(&leaf_cert)?,
            issuer_certificate_pem: pem_string(&root_cert)?,
        };

        Ok((issued, pem_string(&root_cert)?))
    }

    #[test]
    fn collaborative_manifest_round_trip_and_verify() -> Result<()> {
        let signer = DummySigner;
        let roster = vec![
            Participant {
                identifier: "alice".to_string(),
                certificate_fingerprint: "fp-alice".to_string(),
                public_key_der: vec![1, 1, 1],
            },
            Participant {
                identifier: "bob".to_string(),
                certificate_fingerprint: "fp-bob".to_string(),
                public_key_der: vec![2, 2, 2],
            },
            Participant {
                identifier: "carol".to_string(),
                certificate_fingerprint: "fp-carol".to_string(),
                public_key_der: vec![3, 3, 3],
            },
        ];

        let claim_digest = ClaimDigest(Sha256::digest(b"collaborative claim digest").to_vec());
        let partials = roster
            .iter()
            .map(|participant| signer.partial_sign(participant, &claim_digest))
            .collect::<Result<Vec<_>>>()?;
        let aggregate_signature =
            signer.aggregate_signatures(&claim_digest, &roster, &partials)?;
        let aggregate_public_key = signer.aggregate_public_key(&roster)?;
        let profile =
            build_aggregate_certificate_profile(aggregate_public_key.clone(), &roster)?;
        let (issued_certificate, root_pem) = issue_test_collaborative_certificate(&profile)?;
        let root_pem_for_trust = root_pem.clone();
        let root_pem_for_route = root_pem.clone();
        let authorization = AuthorizationPackage {
            claim_digest: claim_digest.clone(),
            aggregate_signature: aggregate_signature.clone(),
            aggregate_certificate_hash: hex::encode(Sha256::digest(
                serde_json::to_vec(&issued_certificate).map_err(other_error)?,
            )),
        };
        let embedded = EmbeddedCollaborativeManifest {
            authorization,
            aggregate_certificate: issued_certificate,
        };

        let source = include_bytes!("../tests/fixtures/IMG_0003.jpg");
        let mut source_stream = std::io::Cursor::new(source.as_slice());
        let mut output_stream = std::io::Cursor::new(Vec::<u8>::new());

        let context =
            Context::new().with_signer(test_signer(crate::crypto::raw_signature::SigningAlg::Ps256));
        let mut builder = Builder::from_context(context).with_definition(simple_manifest_json())?;
        builder.add_assertion_json(COLLABORATIVE_AUTHORIZATION_LABEL, &embedded)?;
        let placeholder = builder.placeholder("image/jpeg")?;
        let offset = write_jpeg_placeholder_stream(
            &placeholder,
            "image/jpeg",
            &mut source_stream,
            &mut output_stream,
            None,
        )?;
        output_stream.set_position(0);
        builder.update_hash_from_stream("image/jpeg", &mut output_stream)?;
        let signed_manifest = builder.sign_embeddable("image/jpeg")?;

        let mut patched_stream = std::io::Cursor::new(Vec::new());
        patch_stream(
            &mut output_stream,
            &mut patched_stream,
            offset as u64,
            placeholder.len() as u64,
            &signed_manifest,
        )
        .map_err(other_error)?;

        patched_stream.set_position(0);
        let reader = Reader::default()
            .with_stream("image/jpeg", &mut patched_stream)?;
        assert!(reader.active_manifest().is_some());

        let extracted = reader
            .active_manifest()
            .expect("active manifest should be present")
            .find_assertion(COLLABORATIVE_AUTHORIZATION_LABEL)?;
        assert_eq!(extracted, embedded);

        let verifier = CollaborativeVerifier::with_trust_store(
            signer,
            TrustStore::from_pem(root_pem_for_trust)?,
        );
        let summary = verifier.verify_embedded(&extracted, &roster)?;
        assert!(summary.verified);
        assert_eq!(summary.approver_ids, vec!["alice", "bob", "carol"]);
        assert_eq!(summary.aggregate_algorithm, aggregate_signature.algorithm);

        let route = verify_issued_certificate_by_route(
            &signer,
            &embedded.aggregate_certificate,
            &roster,
            &[root_pem_for_route],
            true,
        )?;
        assert_eq!(route, CertificateVerificationRoute::Collaborative);

        // Keep the test honest by confirming the embedded certificate profile can be
        // recovered directly as well.
        let recovered = extract_certificate_profile(&embedded.aggregate_certificate)?;
        assert_eq!(recovered, profile);

        Ok(())
    }

    #[test]
    fn standard_certificate_uses_standard_route() -> Result<()> {
        let signer = DummySigner;
        let roster = vec![Participant {
            identifier: "alice".to_string(),
            certificate_fingerprint: "fp-alice".to_string(),
            public_key_der: vec![1, 1, 1],
        }];

        let (issued_certificate, root_pem) = issue_test_standard_certificate()?;
        let route = certificate_verification_route(&issued_certificate)?;
        assert_eq!(route, CertificateVerificationRoute::StandardC2pa);

        let routed = verify_issued_certificate_by_route(
            &signer,
            &issued_certificate,
            &roster,
            &[root_pem],
            true,
        )?;
        assert_eq!(routed, CertificateVerificationRoute::StandardC2pa);

        Ok(())
    }
}
