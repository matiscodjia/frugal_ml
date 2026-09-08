//Spécification du code storage
// N'hesite pas si ma réponse n'est pas précise à rester dessus pour soit me
//redemander de réfléchir plus ou me donner un indice, mais ne sois pas complaisant avec moi
//a) Le trait doit être implémenté pour Scalar lui-même et pour [B; N] où B implémente déjà le trait. Pourquoi cette construction par récurrence plutôt qu'un simple impl<const N: usize> Buffer for [Scalar; N] ? Qu'est-ce que la version récursive te permet d'écrire que l'autre ne permet pas ?
//Reponse a : Pour facilement composer des buffers de buffers, donc des tableaux de tableau
//directement en buffer.
//
//n)Le trait a besoin d'exposer une vue à plat, &[Scalar], sur un buffer qui peut être imbriqué ([[Scalar; 4]; 3]). Quelle opération vas-tu devoir faire pour obtenir cette tranche, et pourquoi le compilateur ne peut-il pas la valider tout seul ?
//Réponse b: Il va falloir que j'établisse un pointer directement sur le tableau 2D qui en mémoire est déjà un
//tableau 1D. Le compilateur ne peut plus valider quand le tableau s'arrête si on lit directement
//en mémoire sans const generiques.
//c) De (b) découle que le trait sera unsafe. Formule l'invariant : qu'est-ce que celui qui écrit un impl Buffer promet exactement ? Deux propriétés, l'une sur la disposition mémoire, l'autre sur une valeur particulière.
//Invariant : On ne doit jamais lire au dela de la valeur du nombres d'elements qu'il y a dans le
//tenseur. On avance toujours d'un pas de f32 en mémoire soit 4 bytes.
//
//Réponse a améliorée: effectivement il faut implémenter un buffer pour tous les types de structs
//(une infinité théorique sur les tailles). La composition récursive permet d'écrire des tableau de
//buffers ce qui par définition même en fait aussi des buffers donc la propriété est stable par
//compositon - Validée
//
//Réponse b améliorée : Il y a apparemment std slice qui renvoie un iterateur mais c'est std; nous
//sommes en no_std donc on est obligé d'utiliser un raw pointer.
//
//Réponse c: Dans une struct l'alignement peut casser cette structure. L'utilisateur doit promettre
//un alignement constant quelque soit la struct qui implémente buffer et pour la seconde valeur je
//ne sais pas. Scalar doit être nullable en tant que struct. ça peut preter à confusion parce que
//Scalar est un type alias de f32.
//
//Réponse b améliorée: Le compilateur ne sait pas en fonction de la struct s'il y a du padding
//d'alignement. DOnc au moment du déréférencement ou de l'intialisation du pointer, on risque de lire des valeurs corrompues (bits de padding) - Validée
//Réponse c améliorée:  L'auteur doit garantir  que le type qui implemente buffer peut être
//zeroable. Un type est zeroable si le motif binaire 0...000 est une représentation légale d'une de
//ses valeurs. Invariant : Pas de padding LEN scalaires strictements contigues et Zeroable le motif
//tout-zero est une valeur B valide - Validée
//
// Safety: Lorsqu'on manipule les types il faut vérifier deux choses à l'implémentation du trait buffer.
// L'utilisateur doit garantir l'absence de padding que le compilateur ne sait pas détecter.
// L'utilisateur doit garantir que le type qui implémente le trait est zeroable et que le motif
// 0..00 est un motif légal. Ce qui permet d'initialiser un buffer à zéro
//
//
use crate::scalar::Scalar;
use core::slice::from_raw_parts;
pub unsafe trait Buffer {
    const LEN: usize;
    fn as_flat(&self) -> &[Scalar];
    fn as_flat_mut(&mut self) -> &mut [Scalar];
    fn zeroed_inline() -> Self;
}
//Fin de séance :
//Résumé
//Difficultés à travailler : tu vas trop vite sur la syntaxe avant d'avoir figé le concept — plusieurs allers-retours (as_flat, as_flat_mut, zeroed) où une forme correcte que tu venais d'établir a été réécrite fausse juste après, signe que la solution n'était pas encore internalisée mais recopiée dans l'instant. Tu confonds encore par endroits les couches du langage : trait vs type concret (self::Buffer au lieu de Self), pointeur brut vs référence vs slice (empilage de *const, &, [T] alors qu'une seule couche suffit), et méthode vs constante associée (une méthode ne peut pas exposer une valeur connue au niveau du type). Sur le fond conceptuel, tu esquives parfois la question posée pour répondre à une question voisine plus confortable (b) sur le padding répondu en "std vs no_std", c) sur zeroable répondu en "reroable" flou) — à surveiller particulièrement, car c'est le signe le plus fiable que le concept n'est pas encore acquis.
//Qualités sur lesquelles t'appuyer : ta première réponse sur la récursivité du buffer (stabilité par composition) était juste et bien formulée du premier coup, sans aide. Tu corriges vite et proprement une fois l'indice donné — la trajectoire as_flat → as_flat_mut a montré que tu retiens la correction précédente pour l'appliquer par analogie, même si l'erreur "self au lieu de &mut self" est repassée une fois. Tu ne fais pas semblant de comprendre : tu dis explicitement "vraiment pas sûr" ou "je ne sais pas" plutôt que de bluffer, ce qui est la condition nécessaire pour que ce mode d'apprentissage fonctionne.
