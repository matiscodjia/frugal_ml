//! The traits every layer implements, and the combinator that chains them.
//!
//! A network is a type: `Then<A, B>` composes two modules, and each trait here
//! lifts an operation from the leaves to the whole tree — forward/backward
//! ([`Module`]), gradient shape ([`params::Params`], plumbing only — implement
//! it to write a custom layer, but nothing consumes it directly), in-place
//! descent ([`Update`]), and the flat views ([`FlatGrads`], [`Perturb`]) that
//! gradient checking needs.

pub mod flat_grads;
pub mod module;
pub mod params;
pub mod perturb;
pub mod sequential;
pub mod update;

pub use flat_grads::FlatGrads;
pub use module::Module;
pub use perturb::Perturb;
pub use sequential::Then;
pub use update::Update;
// `Params` reste accessible via `core::params::Params` (necessaire pour
// implementer un Module custom, puisque `Module: Params`), mais n'est pas
// reexporte ici -- personne ne le nomme directement en usage normal (verifie
// : aucune occurrence hors des `impl Params for ...` qui le fournissent).
