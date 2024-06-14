
const FSST_ESC: u8 = 255;
const _FSST_LEN_BITS: u16 = 12;
const FSST_CODE_BITS: u16 = 9;
// first 256 codes [0,255] are pseudo codes: escaped bytes
const FSST_CODE_BASE: u16 = 256;
const FSST_CODE_MAX: u16 = 1 << FSST_CODE_BITS;
// all bits set: indicating a symbol that has not been assigned a code yet
const FSST_CODE_MASK: u16 = FSST_CODE_MAX - 1;

const FSST_HASH_LOG2SIZE: usize = 10;
const FSST_HASH_PRIME: u64 = 2971215073;
const FSST_SHIFT: usize = 15;
const FSST_HASH: fn(u64) -> u64 = |w| ((w.wrapping_mul(FSST_HASH_PRIME)^((w.wrapping_mul(FSST_HASH_PRIME)))>>FSST_SHIFT));
const MAX_SYMBOL_LENGTH: usize = 8;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::ptr;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use lance_core::Result;

fn fsst_unaligned_load(v: *const u8) -> u64 {
    // compiler will generate efficient code (unaligned load, where possible)
    unsafe { ptr::read_unaligned(v as *const u64) }
}

#[derive(Default, Copy, Clone, PartialEq, Eq)]
struct Symbol {
    // the byte sequence that this symbol stands for
    val: u64, // usually we process it as a num(ber), as this is fast

    // icl = u64 ignoredBits:16,code:12,length:4,unused:32 -- but we avoid exposing this bit-field notation
    icl: u64,  // use a single u64 to be sure "code" is accessed with one load and can be compared with one comparison
}

use std::fmt;

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.val.to_ne_bytes();
        for i in 0..self.length() {
            write!(f, "{}", bytes[i as usize] as char)?;
        }
        write!(f, "\t")?;
        write!(f, "ignoredBits: {}, code: {}, length: {}", self.ignored_bits(), self.code(), self.length())?;
        Ok(())
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol {{ val: {}, icl: {} }}", self.val, self.icl)
    }
}

impl Symbol {
    fn new() -> Self {
        Self { val: 0, icl: FSST_ICL_FREE as u64}
    }

    fn from_char(c: u8, code: u16) -> Self {
        Self { val: c as u64, icl: (1<<28)|(code as u64)<<16|56 } // 56 = 8*7, in a symbol which represents a single 
                                                                  // character, 56 bits are ignored
    }

    fn from_char_slice(input: &[u8]) -> Self {
        let mut s = Self::new();
        s.val = 0;
        let len = if input.len() >= MAX_SYMBOL_LENGTH { MAX_SYMBOL_LENGTH } else { input.len() };
        unsafe { ptr::copy_nonoverlapping(input.as_ptr(), &mut s.val as *mut _ as *mut u8, len); }
        s.set_code_len(FSST_CODE_MAX as u16, len as u32);
        s
    }

    fn set_code_len(&mut self, code: u16, len: u32) {
        self.icl = ((len as u64)<< 28)|((code as u64) << 16)|((8u64.saturating_sub(len as u64))*8);
    }

    fn length(&self) -> u32 {
        (self.icl >> 28) as u32
    }

    fn code(&self) -> u16 {
        ((self.icl >> 16) & FSST_CODE_MASK as u64) as u16
    }

    fn ignored_bits(&self) -> u32 {
        (self.icl & u16::MAX as u64) as u32
    }

    fn first(&self) -> u8 {
        assert!(self.length() >= 1);
        (0xFF & self.val) as u8
    }

    fn first2(&self) -> u16 {
        assert!(self.length() >= 2);
        (0xFFFF & self.val) as u16
    }


    fn hash(&self) -> u64 {
        let v = 0xFFFFFF & self.val;
        (FSST_HASH)(v)
    }

    fn concat(a: Self, b: Self) -> Self {
        let mut s = Self::new();
        let mut length = a.length() + b.length();
        if length > MAX_SYMBOL_LENGTH as u32 {
            length = MAX_SYMBOL_LENGTH as u32;
        }
        s.set_code_len(FSST_CODE_MASK as u16, length);
        s.val = (b.val << (8 * a.length())) | a.val;
        s
    }
}

// Symbol that can be put in a queue, ordered on gain
#[derive(Clone)]
struct QSymbol {
    symbol: Symbol,
    gain: u32, // mutable because gain value should be ignored in find() on unordered_set of QSymbols
}

impl PartialEq for QSymbol {
    fn eq(&self, other: &Self) -> bool {
        self.symbol.val == other.symbol.val && self.symbol.icl == other.symbol.icl
    }
}

impl Ord for QSymbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gain.cmp(&other.gain)
            .then_with(|| other.symbol.val.cmp(&self.symbol.val)) // first order by gain, the by symbol val alphabetically
    }
}

impl PartialOrd for QSymbol {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for QSymbol {}

use std::hash::{Hash, Hasher};

impl Hash for QSymbol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut k = self.symbol.val;
        const M: u64 = 0xc6a4a7935bd1e995;
        const R: u32 = 47;
        let mut h: u64 = 0x8445d61a4e774912 ^ (8u64.wrapping_mul(M));
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
        h ^= h >> R;
        h = h.wrapping_mul(M);
        h ^= h >> R;
        h.hash(state);
    }
}

// we construct FSST symbol tables using a random sample of about 16KB (1<<14) 
const FSST_SAMPLETARGET: usize = 1 << 14;
const FSST_SAMPLEMAXSZ: usize = 2 * FSST_SAMPLETARGET;
const _FSST_SAMPLELINE: usize = 512;

// in the hash table, the icl field contains (low-to-high) ignoredBits:16,code:12,length:4

const FSST_ICL_FREE: u32 = (8 << 28) | ((FSST_CODE_MASK as u32) << 16);

// ignoredBits is (8-length)*8, which is the amount of high bits to zero in the input word before comparing with the hashtable key
//             ..it could of course be computed from len during lookup, but storing it precomputed in some loose bits is faster
//
// the gain field is only used in the symbol queue that sorts symbols on gain

#[derive(Clone)]
struct SymbolTable {
    short_codes: [u16; 65536],
    byte_codes: [u16; 256],
    symbols: [Symbol; FSST_CODE_MAX as usize],
    hash_tab: [Symbol; 1 << FSST_HASH_LOG2SIZE as usize],
    hash_tab_size: usize,
    n_symbols: u16,
    // in a finalized symbol table, symbols are arranged by their symbol length, 
    // in the order of 2, 3, 4, 5, 6, 7, 8, 1, codes < suffix_lim are 2 bytes codes that don't have a longer suffix
    suffix_lim: u16, 
    len_histo: [u8; FSST_CODE_BITS as usize],
}

impl std::fmt::Display for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A FSST SymbolTable after finalize():\n")?;
        write!(f, "n_symbols: {}\n", self.n_symbols)?;
        for i in 0 as usize..self.n_symbols as usize {
            write!(f, "symbols[{}]: {}\n", i, self.symbols[i])?;
        }
        write!(f, "suffix_lim: {}\n", self.suffix_lim)?;
        for i in 0..FSST_CODE_BITS {
            write!(f, "len_histo[{}]: {}\n", i, self.len_histo[i as usize])?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A FSST SymbolTable before finalize():\n")?;
        write!(f, "n_symbols: {}\n", self.n_symbols)?;
        for i in FSST_CODE_BASE as usize..FSST_CODE_BASE as usize + self.n_symbols as usize {
            write!(f, "symbols[{}]: {}\n", i, self.symbols[i])?;
        }
        write!(f, "suffix_lim: {}\n", self.suffix_lim)?;
        for i in 0..FSST_CODE_BITS {
            write!(f, "len_histo[{}]: {}\n", i, self.len_histo[i as usize])?;
        }
        Ok(())
    }
}

impl SymbolTable {
    fn new() -> Self {
        let mut symbols = [Symbol::new(); FSST_CODE_MAX as usize];
        for i in 0..256 {
            symbols[i] = Symbol::from_char(i as u8, i as u16);
        }
        let unused = Symbol::from_char(0, FSST_CODE_MASK as u16);
        for i in 256..FSST_CODE_MAX {
            symbols[i as usize] = unused;
        }
        let s = Symbol::new();
        let hash_tab = [s; 1 << FSST_HASH_LOG2SIZE];
        let mut byte_codes = [0; 256];
        for i in 0..256 {
            byte_codes[i] = i as u16;
        }
        let mut short_codes = [FSST_CODE_MASK; 65536];
        for i in 0..=65535 as u16 {
            // in a empty short_code, the lower 8-bit is the escape code 255, the higher 8-bit is the index's higher 8-bit
            short_codes[i as usize] = i & 0xFF; 
        }
        Self {
            short_codes,
            byte_codes,
            symbols,
            hash_tab,
            hash_tab_size: 1 << FSST_HASH_LOG2SIZE,
            n_symbols: 0,
            suffix_lim: FSST_CODE_MAX as u16,
            len_histo: [0; FSST_CODE_BITS as usize],
        }
    } 

    fn clear(&mut self) {
        for i in 0..256 {
            self.symbols[i] = Symbol::from_char(i as u8, i as u16);
        }
        let unused = Symbol::from_char(0, FSST_CODE_MASK as u16);
        for i in 256..FSST_CODE_MAX {
            self.symbols[i as usize] = unused;
        }
        let s = Symbol::new();
        for i in 0..256 {
            self.byte_codes[i] = i as u16;
        }
        for i in 0..=65535 as u16 {
            self.short_codes[i as usize] = i & 0xFF;
        }
        for i in 0..1 << FSST_HASH_LOG2SIZE {
            self.hash_tab[i] = s;
        }
        for i in 0..FSST_CODE_BITS as usize {
            self.len_histo[i] = 0;
        }
        self.n_symbols = 0;

    }

    fn hash_insert(&mut self, s: Symbol) -> bool {
        let idx = (s.hash() & (self.hash_tab_size as u64 - 1)) as usize;
        let taken = self.hash_tab[idx].icl < FSST_ICL_FREE as u64;
        if taken {
            return false; // collision in hash table
        }
        self.hash_tab[idx].icl = s.icl;
        self.hash_tab[idx].val = s.val & (u64::MAX >> (s.ignored_bits()));
        true
    }

    fn add(&mut self, mut s: Symbol) -> bool {
        assert!(FSST_CODE_BASE as u16 + self.n_symbols < FSST_CODE_MAX as u16);
        let len = s.length();
        s.set_code_len(FSST_CODE_BASE as u16 + self.n_symbols, len);
        if len == 1 {
            self.byte_codes[s.first() as usize] = FSST_CODE_BASE + self.n_symbols;
        } else if len == 2 {
            self.short_codes[s.first2() as usize] = FSST_CODE_BASE + self.n_symbols;
        } else if !self.hash_insert(s) {
            return false;
        }
        self.symbols[(FSST_CODE_BASE + self.n_symbols) as usize] = s.clone();
        self.n_symbols += 1;
        self.len_histo[(len - 1) as usize] += 1;
        true
    }

    fn find_longest_symbol(&self, s: Symbol) -> u16 {
        let idx = (s.hash() & (FSST_HASH_LOG2SIZE as u64 - 1)) as usize;
        //println!("self.hash_tab[idx].icl: {:?}", self.hash_tab[idx].icl);
        if self.hash_tab[idx].icl == s.icl && self.hash_tab[idx].val == s.val {
            return ((self.hash_tab[idx].icl >> 16) & FSST_CODE_MASK as u64) as u16; // matched a long symbol 
        }
        if s.length() >= 2 {
            let code =  self.short_codes[s.first2() as usize] & FSST_CODE_MASK;
            if !is_escape_code(code) {
                return code; 
            }
        }
        self.byte_codes[s.first() as usize] & FSST_CODE_MASK
    }

    fn find_longest_symbol_from_char_slice(&self, input: &[u8]) -> u16 {
        self.find_longest_symbol(Symbol::from_char_slice(input))
    }

   // rationale for finalize:
   // - during symbol table construction, we may create more than 256 codes, but bring it down to max 255 in the last makeTable()
   //   consequently we needed more than 8 bits during symbol table contruction, but can simplify the codes to single bytes in finalize()
   //   (this feature is in fact lo longer used, but could still be exploited: symbol construction creates no more than 255 symbols in each pass)
   // - we not only reduce the amount of codes to <255, but also *reorder* the symbols and renumber their codes, for higher compression perf.
   //   we renumber codes so they are grouped by length, to allow optimized scalar string compression (byteLim and suffixLim optimizations). 
   // - we make the use of byteCode[] no longer necessary by inserting single-byte codes in the free spots of shortCodes[]
   //   Using shortCodes[] only makes compression faster. When creating the symbolTable, however, using shortCodes[] for the single-byte
   //   symbols is slow, as each insert touches 256 positions in it. This optimization was added when optimizing symbolTable construction time.
   //
   // In all, we change the layout and coding, as follows..
   //
   // before finalize(): 
   // - The real symbols are symbols[256..256+nSymbols>. As we may have nSymbols > 255
   // - The first 256 codes are pseudo symbols (all escaped bytes)
   //
   // after finalize(): 
   // - table layout is symbols[0..nSymbols>, with nSymbols < 256. 
   // - Real codes are [0,nSymbols>. 8-th bit not set. 
   // - Escapes in shortCodes have the 8th bit set (value: 256+255=511). 255 because the code to be emitted is the escape byte 255
   // - symbols are grouped by length: 2,3,4,5,6,7,8, then 1 (single-byte codes last)
   // the two-byte codes are split in two sections: 
   // - first section contains codes for symbols for which there is no longer symbol (no suffix). It allows an early-out during compression
   //
   // finally, shortCodes[] is modified to also encode all single-byte symbols (hence byteCodes[] is not required on a critical path anymore).
   //
    fn finalize(&mut self) {
        assert!(self.n_symbols <= 255);
        let mut new_code: [u8; 256] = [0; 256];
        let mut rsum: [u8; 8] = [0; 8];
        let byte_lim = self.n_symbols - self.len_histo[0] as u16;

        rsum[0] = byte_lim as u8; // 1-byte codes are highest
        for i in 1..7 {
            rsum[i + 1] = rsum[i] + self.len_histo[i] as u8;
        }

        let mut suffix_lim = 0;
        let mut j = rsum[2];
        for i in 0..self.n_symbols {
            let mut s1 = self.symbols[(FSST_CODE_BASE + i) as usize];
            let len = s1.length();
            let mut opt = if len == 2 { self.n_symbols } else { 0 };
            if opt != 0 {
                let first2 = s1.first2();
                for k in 0..opt {
                    let s2 = self.symbols[(FSST_CODE_BASE + k)as usize];
                    if k != i && s2.length() > 1 && first2 == s2.first2() {
                        opt = 0;
                    }
                }
                new_code[i as usize] = if opt != 0 { suffix_lim += 1; suffix_lim - 1 } else { j -= 1; j };
            } else {
                new_code[i as usize] = rsum[(len - 1) as usize];
                rsum[(len - 1) as usize] += 1;
            }
            s1.set_code_len(new_code[i as usize] as u16, len);
            self.symbols[new_code[i as usize] as usize] = s1;
        }

        for i in 0..256 {
            if (self.byte_codes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE {
                self.byte_codes[i] = new_code[(self.byte_codes[i] & 0xFF) as usize] as u16;
            } else {
                self.byte_codes[i] = 511;
            }
        }

        for i in 0..65536 {
            if (self.short_codes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE {
                self.short_codes[i] = new_code[(self.short_codes[i] & 0xFF) as usize] as u16;
            } else {
                self.short_codes[i] = self.byte_codes[(i & 0xFF) as usize];
            }
        }

        for i in 0..self.hash_tab_size {
            if self.hash_tab[i].icl < FSST_ICL_FREE as u64{
                self.hash_tab[i] = self.symbols[new_code[(self.hash_tab[i].code() & 0xFF) as usize] as usize];
            }
        }
        self.suffix_lim = suffix_lim as u16;
    }
}

#[derive(Clone)]
struct Counters {
    count1: Vec::<u16>,
    count2: Vec::<Vec::<u16>>,
}

impl Counters {
    fn new() -> Self {
        Self {
            count1: vec![0; FSST_CODE_MAX as usize],
            count2: vec![vec![0; FSST_CODE_MAX as usize]; FSST_CODE_MAX as usize],
        }
    }

    /*
    fn count1_set(&mut self, pos1: usize, val: u16) {
        self.count1[pos1] = val;
    }
    */
    fn count1_inc(&mut self, pos1: u16) {
        self.count1[pos1 as usize] = self.count1[pos1 as usize].saturating_add(1);
    }

    fn count2_inc(&mut self, pos1: usize, pos2: usize) {
        self.count2[pos1][pos2] = self.count2[pos1][pos2].saturating_add(1);
    }

    fn count1_get(&self, pos1: usize) -> u16 {
        self.count1[pos1]
    }

    fn count2_get(&self, pos1: usize, pos2: usize) -> u16 {
        self.count2[pos1][pos2]
    }

    /* 
    fn clear(&self) {
        Self {
            count1: vec![0; FSST_CODE_MAX as usize],
            count2: vec![vec![0; FSST_CODE_MAX as usize]; FSST_CODE_MAX as usize],
        };
    }*/
}

fn is_escape_code(pos: u16) -> bool {
    pos < FSST_CODE_BASE as u16
}

fn make_sample(strs: &[u8], offsets: &[i32]) -> (Vec<u8>, Vec<i32>) {
    let total_size = strs.len();
    if total_size <= FSST_SAMPLETARGET {
        return (strs.to_vec(), offsets.to_vec());
    }
    let mut sample = Vec::with_capacity(FSST_SAMPLEMAXSZ);
    let mut sample_offsets: Vec<i32> = Vec::new();

    sample_offsets.push(0);
    let mut rng = StdRng::from_entropy();
    while sample.len() < FSST_SAMPLETARGET {
        let rand_num = rng.gen_range(0..offsets.len()) % (offsets.len() - 1);
        sample.extend_from_slice(&strs[offsets[rand_num] as usize ..offsets[rand_num + 1] as usize]);
        sample_offsets.push(sample.len() as i32);
    }
    sample_offsets.push(sample.len() as i32);
    return (sample, sample_offsets);
}

fn build_symbol_table(strs: Vec<u8>, offsets: Vec<i32>) -> Box<SymbolTable> {
    let mut st = SymbolTable::new();
    let mut best_table = SymbolTable::new();
    let mut best_gain = -(FSST_SAMPLEMAXSZ as i32); // worst case (everything exception)

    // Compress sample, and compute (pair-)frequencies
    let compress_count = |st: &mut SymbolTable, sample_frac: usize| -> (Box<Counters>, i32) {
        // a random number between 1 and 128
        let _rnd128 = |i: usize| -> usize { 
            1 + ((FSST_HASH((i as u64 + 1) * sample_frac as u64)&127) as usize) 
        };
        let mut gain:i32 = 0;
        let mut counters = Counters::new();

        for i in 1..offsets.len() {
            /* 
            // this is commented out during development
            if sample_frac < 128 && rnd128(i) > sample_frac {
                continue;
            }*/
            if offsets[i] == offsets[i-1] {
                continue;
            }
            let word = &strs[offsets[i-1] as usize..offsets[i] as usize];

            let mut curr = 0;
            let mut curr_code;
            assert!(word.len() > 0);
            let mut prev_code = st.find_longest_symbol_from_char_slice(&word[curr..]);
            curr += st.symbols[prev_code as usize].length() as usize;
            gain += st.symbols[prev_code as usize].length() as i32 - (1 + is_escape_code(prev_code) as i32);
            while curr < word.len() {
                // Count single symbol (i.e. an option is not extending it)
                counters.count1_inc(prev_code);
                if st.symbols[prev_code as usize].length() != 1 {
                    counters.count1_inc(word[curr] as u16);
                }

                let symbol_len;
                if word.len() > 7 && curr < word.len() - 7 {
                    let mut this_64_bit_word: u64 = fsst_unaligned_load(word[curr..].as_ptr());
                    let code = this_64_bit_word & 0xFFFFFF;
                    let idx = FSST_HASH(code) as usize & (st.hash_tab_size - 1);
                    let s: Symbol = st.hash_tab[idx];
                    let short_code = st.short_codes[(this_64_bit_word & 0xFFFF) as usize] & FSST_CODE_MASK;
                    this_64_bit_word &= 0xFFFFFFFFFFFFFFFF >> s.icl as u8;
                    if (s.icl < FSST_ICL_FREE as u64) & (s.val == this_64_bit_word) {
                        curr_code = s.code();
                        symbol_len = s.length();
                    } else if short_code < (255 << 8) {
                        curr_code = short_code;
                        symbol_len = 2;
                    } else {
                        curr_code = st.byte_codes[(this_64_bit_word & 0xFF) as usize] & FSST_CODE_MASK;
                        symbol_len = 1;
                    }
                } else {
                    curr_code = st.find_longest_symbol(Symbol::from_char_slice(&word[curr..]));
                    symbol_len = st.symbols[curr_code as usize].length();
                }
                gain += symbol_len as i32 - (1 + is_escape_code(curr_code) as i32);
                if sample_frac < 128 { // no need to count pairs in final round
                    // consider the symbol that is the concatenation of the last two symbols
                    counters.count2_inc(prev_code as usize, curr_code as usize);
                    if symbol_len > 1 {
                        counters.count2_inc(prev_code as usize, word[curr] as usize);
                    }
                }
                curr += symbol_len as usize;
                prev_code = curr_code;
            }
            counters.count1_inc(prev_code);
        }
        (Box::new(counters), gain)
    };

    let make_table = |st: &mut SymbolTable, counters: &Counters, sample_frac: usize| {
        let mut candidates: HashSet<QSymbol> = HashSet::new();

        let add_or_inc = |cands: &mut HashSet<QSymbol>, s: Symbol, count: u64| {
            if count < (5 * sample_frac as u64) / 128 {
                return;
            }
            let mut q = QSymbol {
                symbol: s,
                gain: (count * s.length() as u64) as u32,
            };
            if let Some(old_q) = cands.get(&q) {
                q.gain += old_q.gain;
                cands.remove(&old_q.clone());
            } 
            cands.insert(q);
        };

        // add candidate symbols based on counted frequencies
        for pos1 in 0..FSST_CODE_BASE as usize + st.n_symbols as usize {
            let cnt1 = counters.count1_get(pos1);
            if cnt1 == 0 {
                continue;
            }
            // heuristic: promoting single-byte symbols (*8) helps reduce exception rates and increases [de]compression speed
            let s1 = st.symbols[pos1];
            add_or_inc(&mut candidates, s1, if s1.length() == 1 { 8 } else { 1 } * cnt1 as u64);
            if sample_frac >= 128 ||
                s1.length() == MAX_SYMBOL_LENGTH as u32 {
                continue;
            }
            for pos2 in 0..FSST_CODE_BASE as usize + st.n_symbols as usize {
                let cnt2 = counters.count2_get(pos1, pos2);
                if cnt2 == 0 {
                    continue;
                }

                // create a new symbol
                let s2 = st.symbols[pos2];
                let s3 = Symbol::concat(s1, s2);
                add_or_inc(&mut candidates, s3, cnt2 as u64);
            }
        }
        let mut pq: BinaryHeap<QSymbol> = BinaryHeap::new();
        for q in &candidates {
            pq.push(q.clone());
        }

        // Create new symbol map using best candidates
        st.clear();
        while st.n_symbols < 255 && !pq.is_empty() {
            let q = pq.pop().unwrap();
            st.add(q.symbol);
        }
    };

    for frac in [8, 38, 68, 98, 128] { // we do 5 rounds (sampleFrac=8,38,68,98,128)
        let (this_counter, gain ) = compress_count(&mut st, frac);
        if gain >= best_gain { // a new best solution!
            best_gain = gain;
            best_table = st.clone();
        } 
        make_table(&mut st, &this_counter, frac);
    }
    best_table.finalize(); // renumber codes for more efficient compression
    return Box::new(best_table);
}    

// two phases of compression, before and after optimize():
//
// (1) to encode values we probe (and maintain) three datastructures:
// - u16 byteCodes[256] array at the position of the next byte  (s.length==1)
// - u16 shortCodes[65536] array at the position of the next twobyte pattern (s.length==2)
// - Symbol hashtable[1024] (keyed by the next three bytes, ie for s.length>2), 
// this search will yield a u16 code, it points into Symbol symbols[]. You always find a hit, because the first 256 codes are 
// pseudo codes representing a single byte these will become escapes)
//
// (2) when we finished looking for the best symbol table we call optimize() to reshape it:
// - it renumbers the codes by length (first symbols of length 2,3,4,5,6,7,8; then 1 (starting from byteLim are symbols of length 1)
//   length 2 codes for which no longer suffix symbol exists (< suffixLim) come first among the 2-byte codes 
//   (allows shortcut during compression)
// - for each two-byte combination, in all unused slots of shortCodes[], it enters the byteCode[] of the symbol corresponding 
//   to the first byte (if such a single-byte symbol exists). This allows us to just probe the next two bytes (if there is only one
//   byte left in the string, there is still a terminator-byte added during compression) in shortCodes[]. That is, byteCodes[]
//   and its codepath is no longer required. This makes compression faster. The reason we use byteCodes[] during symbolTable construction
//   is that adding a new code/symbol is expensive (you have to touch shortCodes[] in 256 places). This optimization was
//   hence added to make symbolTable construction faster.
//
// this final layout allows for the fastest compression code, only currently present in compressBulk

fn compress_bulk(st: &SymbolTable, strs: &[u8], offsets: &[i32], out: &mut Vec<u8>, out_offsets: &mut Vec<i32>, out_pos: &mut usize, out_offsets_len: &mut usize) {
    let suffix_lim = st.suffix_lim;
    let byte_lim = st.n_symbols - st.len_histo[0] as u16;
    let mut out_curr = *out_pos;

    let mut compress = |buf: &[u8], in_end: usize, out_curr: &mut usize| { 
        let mut in_curr = 0;
        while in_curr < in_end {
            let word = fsst_unaligned_load(buf[in_curr..].as_ptr());
            let code = st.short_codes[(word & 0xFFFF) as usize];
            if code < suffix_lim {
                out[*out_curr] = code as u8;
                *out_curr += 1;
                in_curr += 2;
            } else {
                let code_first_3byte= word & 0xFFFFFF;
                let idx = FSST_HASH(code_first_3byte) as usize & (st.hash_tab_size - 1);
                let s = st.hash_tab[idx];
                out[*out_curr + 1] = (word & 0xFF) as u8;
                let word2 = word & (0xFFFFFFFFFFFFFFFF as u64 >> s.ignored_bits());
                if s.icl != FSST_ICL_FREE as u64 && s.val == word2 {
                    out[*out_curr] = s.code() as u8;
                    *out_curr += 1;
                    in_curr += s.length() as usize;
                } else if code < byte_lim {
                    out[*out_curr] = code as u8;
                    *out_curr += 1;
                    in_curr += 2;
                } else {
                    // 1 byte code or miss
                    out[*out_curr] = code as u8;
                    *out_curr += 1 + ((code & FSST_CODE_BASE) >> 8) as usize;
                    in_curr += 1;
                }
            }
        }
    };

    for i in 1..offsets.len() {
        let mut in_curr = offsets[i - 1] as usize;
        let end_curr = offsets[i] as usize;
        let mut buf: [u8; 520] = [0; 520]; // +8 sentinel is to avoid 8-byte unaligned-loads going beyond 511 out-of-bounds
        while in_curr < end_curr {
            let in_end = std::cmp::min(in_curr + 511, end_curr);
            {
                buf[..in_end - in_curr].copy_from_slice(&strs[in_curr..in_end]);
            }
            compress(&buf, in_end - in_curr, &mut out_curr);
            in_curr = in_end;
        }
        out_offsets[i] = out_curr as i32; 
    }
    *out_pos = out_curr;
    *out_offsets_len = offsets.len();
}

fn decompress_bulk(st: &SymbolTable, compressed_strs: &[u8], offsets: &[i32], out: &mut Vec<u8>, out_offsets: &mut Vec<i32>, out_pos: &mut usize, out_offsets_len: &mut usize) -> Result<()> {
    let mut decompress = |mut in_curr: usize, in_end: usize, out_curr: &mut usize| {
        let mut prev_esc = false;
        while in_curr < in_end {
            if prev_esc {
                out[*out_curr] = compressed_strs[in_curr];
                *out_curr += 1;
                prev_esc = false;
            } else {
                let code = compressed_strs[in_curr];
                if code == FSST_ESC {
                    prev_esc = true;
                } else {
                    let s = st.symbols[code as usize];
                    let len = s.length();
                    out[*out_curr..*out_curr + len as usize].copy_from_slice(&s.val.to_ne_bytes()[..len as usize]);
                    *out_curr += len as usize;
                }
            }
            in_curr += 1;
        }
    };
    let mut out_curr = *out_pos;
    out_offsets[0] = 0;
    for i in 1..offsets.len() {
        let in_curr = offsets[i - 1] as usize;
        let in_end = offsets[i] as usize;
        decompress(in_curr, in_end, &mut out_curr);
        out_offsets[i] = out_curr as i32;
    }
    *out_pos = out_curr;
    *out_offsets_len = offsets.len();
    Ok(())
}

pub fn fsst_compress(_input_buf: &[u8], _input_offsets_buf: &[i32], _output_buf: &mut Vec<u8>, _output_offsets_buf: &mut Vec<i32>) -> Result<()> {
    Ok(())
}

pub fn fsst_decompress(_input_buf: &[u8], _input_offsets_buf: &[i32], _output_buf: &mut Vec<u8>, _output_offsets_buf: &mut Vec<i32>) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use lance_datagen::ByteCount;
    use rand_xoshiro;
    use rand::Rng;

    use crate::encodings::compression_algo::fsst::*;

    const TEST_PARAGRAPH: &str = "ACT I. Scene I.
    Elsinore. A platform before the Castle.
    
    Enter two Sentinels-[first,] Francisco, [who paces up and down
    at his post; then] Bernardo, [who approaches him].
    
        Ber. Who's there.?
        Fran. Nay, answer me. Stand and unfold yourself.
        Ber. Long live the King!
        Fran. Bernardo?
        Ber. He.
        Fran. You come most carefully upon your hour.
        Ber. 'Tis now struck twelve. Get thee to bed, Francisco.
        Fran. For this relief much thanks. 'Tis bitter cold,
        And I am sick at heart.
        Ber. Have you had quiet guard?
        Fran. Not a mouse stirring.
        Ber. Well, good night.
        If you do meet Horatio and Marcellus,
        The rivals of my watch, bid them make haste.
        Enter Horatio and Marcellus.  

        Fran. I think I hear them. Stand, ho! Who is there?
        Hor. Friends to this ground.
        Mar. And liegemen to the Dane.
        Fran. Give you good night.
        Mar. O, farewell, honest soldier.
            Who hath reliev'd you?
        Fran. Bernardo hath my place.
            Give you good night.                                   Exit.
        Mar. Holla, Bernardo!
        Ber. Say-
            What, is Horatio there ?
        Hor. A piece of him.
        Ber. Welcome, Horatio. Welcome, good Marcellus.
        Mar. What, has this thing appear'd again to-night?
        Ber. I have seen nothing.
        Mar. Horatio says 'tis but our fantasy,
            And will not let belief take hold of him
            Touching this dreaded sight, twice seen of us.
            Therefore I have entreated him along,  
            With us to watch the minutes of this night,
            That, if again this apparition come,
            He may approve our eyes and speak to it.
        Hor. Tush, tush, 'twill not appear.
        Ber. Sit down awhile,
            And let us once again assail your ears,
            That are so fortified against our story,
            What we two nights have seen.
        Hor. Well, sit we down,
            And let us hear Bernardo speak of this.
        Ber. Last night of all,
            When yond same star that's westward from the pole
            Had made his course t' illume that part of heaven
            Where now it burns, Marcellus and myself,
            The bell then beating one-

                                Enter Ghost.

        Mar. Peace! break thee off! Look where it comes again!
        Ber. In the same figure, like the King that's dead.  
        Mar. Thou art a scholar; speak to it, Horatio.
        Ber. Looks it not like the King? Mark it, Horatio.
        Hor. Most like. It harrows me with fear and wonder.
        Ber. It would be spoke to.
        Mar. Question it, Horatio.
        Hor. What art thou that usurp'st this time of night
            Together with that fair and warlike form
            In which the majesty of buried Denmark
            Did sometimes march? By heaven I charge thee speak!
        Mar. It is offended.
        Ber. See, it stalks away!
        Hor. Stay! Speak, speak! I charge thee speak!
                                                            Exit Ghost.
        Mar. 'Tis gone and will not answer.
        Ber. How now, Horatio? You tremble and look pale.
            Is not this something more than fantasy?
            What think you on't?
        Hor. Before my God, I might not this believe
            Without the sensible and true avouch
            Of mine own eyes.  
        Mar. Is it not like the King?
        Hor. As thou art to thyself.
            Such was the very armour he had on
            When he th' ambitious Norway combated.
            So frown'd he once when, in an angry parle,
            He smote the sledded Polacks on the ice.
            'Tis strange.
        Mar. Thus twice before, and jump at this dead hour,
            With martial stalk hath he gone by our watch.
        Hor. In what particular thought to work I know not;
            But, in the gross and scope of my opinion,
            This bodes some strange eruption to our state.
        Mar. Good now, sit down, and tell me he that knows,
            Why this same strict and most observant watch
            So nightly toils the subject of the land,
            And why such daily cast of brazen cannon
            And foreign mart for implements of war;
            Why such impress of shipwrights, whose sore task
            Does not divide the Sunday from the week.
            What might be toward, that this sweaty haste  
            Doth make the night joint-labourer with the day?
            Who is't that can inform me?
        Hor. That can I.
            At least, the whisper goes so. Our last king,
            Whose image even but now appear'd to us,
            Was, as you know, by Fortinbras of Norway,
            Thereto prick'd on by a most emulate pride,
            Dar'd to the combat; in which our valiant Hamlet
            (For so this side of our known world esteem'd him)
            Did slay this Fortinbras; who, by a seal'd compact,
            Well ratified by law and heraldry,
            Did forfeit, with his life, all those his lands
            Which he stood seiz'd of, to the conqueror;
            Against the which a moiety competent
            Was gaged by our king; which had return'd
            To the inheritance of Fortinbras,
            Had he been vanquisher, as, by the same comart
            And carriage of the article design'd,
            His fell to Hamlet. Now, sir, young Fortinbras,
            Of unimproved mettle hot and full,  
            Hath in the skirts of Norway, here and there,
            Shark'd up a list of lawless resolutes,
            For food and diet, to some enterprise
            That hath a stomach in't; which is no other,
            As it doth well appear unto our state,
            But to recover of us, by strong hand
            And terms compulsatory, those foresaid lands
            So by his father lost; and this, I take it,
            Is the main motive of our preparations,
            The source of this our watch, and the chief head
            Of this post-haste and romage in the land.
        Ber. I think it be no other but e'en so.
            Well may it sort that this portentous figure
            Comes armed through our watch, so like the King
            That was and is the question of these wars.
        Hor. A mote it is to trouble the mind's eye.
            In the most high and palmy state of Rome,
            A little ere the mightiest Julius fell,
            The graves stood tenantless, and the sheeted dead
            Did squeak and gibber in the Roman streets;  
            As stars with trains of fire, and dews of blood,
            Disasters in the sun; and the moist star
            Upon whose influence Neptune's empire stands
            Was sick almost to doomsday with eclipse.
            And even the like precurse of fierce events,
            As harbingers preceding still the fates
            And prologue to the omen coming on,
            Have heaven and earth together demonstrated
            Unto our climature and countrymen.";

    #[test_log::test(tokio::test)]
    async fn test_symbol_new() {
        let st = SymbolTable::new();
        assert!(st.n_symbols == 0);
        for i in 0..=255 as u8 {
            assert!(st.symbols[i as usize] == Symbol::from_char(i, i as u16));
        }
        let s = Symbol::from_char(1, 1);
        assert!(s == st.symbols[1]);
        for i in 0..1 << FSST_HASH_LOG2SIZE {
            assert!(st.hash_tab[i] == Symbol::new());
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_symbol_from_slice() {
        let hello_str = "hello";
        let symbol_hello = Symbol::from_char_slice(hello_str.as_bytes());
        assert!(symbol_hello.length() == hello_str.len() as u32);
        assert!(symbol_hello.ignored_bits() == 24); // 8 - 5 = 3
        for i in 0..hello_str.len() {
            assert!(symbol_hello.val.to_ne_bytes()[i] == hello_str.as_bytes()[i]);
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_symbol_add() {
        let mut st = SymbolTable::new();
        let hello_str = "hello";
        st.add(Symbol::from_char_slice(hello_str.as_bytes()));
        let symbol_hello = Symbol::from_char_slice(hello_str.as_bytes());
        assert!(st.symbols[FSST_CODE_BASE as usize].length() == symbol_hello.length());
        assert!(st.symbols[FSST_CODE_BASE as usize].ignored_bits() == symbol_hello.ignored_bits());
        assert!(st.symbols[FSST_CODE_BASE as usize].val == symbol_hello.val);
        assert!(st.n_symbols == 1);
        assert!(false == st.add(symbol_hello));
        assert!(st.n_symbols == 1);
        let world_str = "world";
        let symbol_world = Symbol::from_char_slice(world_str.as_bytes());
        st.add(symbol_world);
        assert!(st.n_symbols == 2);
        assert!(st.symbols[FSST_CODE_BASE as usize + 1].length() == symbol_world.length());
        assert!(st.symbols[FSST_CODE_BASE as usize + 1].ignored_bits() == symbol_world.ignored_bits());
        assert!(st.symbols[FSST_CODE_BASE as usize + 1].val == symbol_world.val);
        let us_str = "us";
        let us_symbol = Symbol::from_char_slice(us_str.as_bytes());
        st.add(us_symbol);
        assert!(st.n_symbols == 3);
        assert!(st.symbols[FSST_CODE_BASE as usize + 2].val == us_symbol.val);
        assert!(st.symbols[FSST_CODE_BASE as usize + 2].ignored_bits() == us_symbol.ignored_bits());
        assert!(st.symbols[FSST_CODE_BASE as usize + 2].length() == us_symbol.length());
        let short_codes_idx: usize = u16::from_ne_bytes([us_str.as_bytes()[0], us_str.as_bytes()[1]]) as usize;
        let code_in_short_codes = st.short_codes[short_codes_idx];
        assert!(code_in_short_codes as usize == (FSST_CODE_BASE + 2) as usize);
        let x_str = "x";
        let x_symbol = Symbol::from_char_slice(x_str.as_bytes());
        st.add(x_symbol);
        assert!(st.n_symbols == 4);
        assert!(st.symbols[FSST_CODE_BASE as usize + 3].val == x_symbol.val);
        assert!(st.symbols[FSST_CODE_BASE as usize + 3].ignored_bits() == x_symbol.ignored_bits());
        assert!(st.symbols[FSST_CODE_BASE as usize + 3].length() == x_symbol.length());
        assert!(st.symbols[code_in_short_codes as usize].val == us_symbol.val);
    }

    #[test_log::test(tokio::test)]
    async fn test_make_sample() {
        // to test make_sample, uncomment this block
        /* 
        let words: Vec<&str> = TEST_PARAGRAPH.split_whitespace().collect();
        let string_array = StringArray::from(words.clone());
        let (sample, sample_offsets) = make_sample(string_array.value_data(), string_array.value_offsets());
        for i in 1..sample_offsets.len() {
            let s = &sample[sample_offsets[i-1] as usize..sample_offsets[i] as usize];
            assert!(words[i-1].as_bytes() == s);
        }
        let paragraph2 = TEST_PARAGRAPH.to_string().repeat(50);
        assert!(paragraph2.len() >= FSST_SAMPLETARGET, "Size of paragraph2 is not greater than 16KB");

        let words2: Vec<&str> = paragraph2.split_whitespace().collect();
        let string_array2 = StringArray::from(words2);
        let (sample2, sample_offsets2) = make_sample(string_array2.value_data(), string_array2.value_offsets());
        assert!(sample2.len() >= FSST_SAMPLETARGET);
        for i in 1..sample_offsets2.len() {
            let s = &sample2[sample_offsets2[i-1] as usize..sample_offsets2[i] as usize];
            println!("{:?}", std::str::from_utf8(s));
        }*/
    }

    #[test_log::test(tokio::test)]
    async fn test_build_symbol_table() {
        // to test build_symbol_table, comment out line 543-546 and uncomment these
        /* 
        let words = vec!["hello", "world", "boston", "alaska"];
        let string_array = StringArray::from(words);
        let st = *build_symbol_table(string_array.values().to_vec(), string_array.value_offsets().to_vec());
        println!("{}", st);
        */
    }

    #[test_log::test(tokio::test)]
    async fn test_compress_bulk() {

        // to test compress_bulk, uncomment this block
        /* 
        let words = vec!["world", "world", "boston", "alaska"];
        let string_array = StringArray::from(words);
        let st = *build_symbol_table(string_array.values().to_vec(), string_array.value_offsets().to_vec());
        let mut output_buffer : Vec<u8> = vec![0; 1024];
        let mut offset_buffer : Vec<i32> = vec![0; 1024];
        let mut out_pos = 0;
        let mut out_offsets_len = 0;
        compress_bulk(&st, string_array.values(), string_array.value_offsets(), &mut output_buffer, &mut offset_buffer, &mut out_pos, &mut out_offsets_len);
        for i in 1..out_offsets_len {
            let s = &output_buffer[offset_buffer[i-1] as usize..offset_buffer[i] as usize];
            println!("{:?}", s);
        }
        let words2 = vec!["hello", "world", "boston", "alaska"];
        let string_array2 = StringArray::from(words2);
        let st2 = *build_symbol_table(string_array2.values().to_vec(), string_array2.value_offsets().to_vec());
        println!("st2: {}", st2);
        let mut output_buffer2 : Vec<u8> = vec![0; 1024];
        let mut offset_buffer2 : Vec<i32> = vec![0; 1024];
        let mut out_pos2 = 0;
        let mut out_offsets_len2 = 0;
        let _ = compress_bulk(&st2, string_array2.values(), string_array2.value_offsets(), &mut output_buffer2, &mut offset_buffer2, &mut out_pos2, &mut out_offsets_len2);
        println!("{:?}", offset_buffer2);
        println!("offset_buffer.len(): {:?}", offset_buffer2.len());
        println!("offset_len: {:?}", out_offsets_len2);
        for i in 1..out_offsets_len2 {
            let s = &output_buffer2[offset_buffer2[i-1] as usize..offset_buffer2[i] as usize];
            println!("{:?}", s);
        }*/
    }

    #[test_log::test(tokio::test)]
    async fn test_fsst() {
        let words = vec!["world", "world", "boston", "alaska"];
        let string_array = StringArray::from(words);
        let st = *build_symbol_table(string_array.values().to_vec(), string_array.value_offsets().to_vec());
        let mut compress_output_buffer: Vec<u8> = vec![0; 16 * 1024];
        let mut compress_offset_buffer: Vec<i32> = vec![0; 16 * 1024];
        let mut compress_out_buf_pos = 0;
        let mut compress_out_offsets_len = 0;
        compress_bulk(&st, string_array.values(), string_array.value_offsets(), &mut compress_output_buffer, &mut compress_offset_buffer, &mut compress_out_buf_pos, & mut compress_out_offsets_len);
        assert!(compress_out_buf_pos < string_array.values().len());
        compress_output_buffer.resize(compress_out_buf_pos, 0);
        compress_offset_buffer.resize(compress_out_offsets_len, 0);
        let mut decompressed_output: Vec<u8> = vec![0; 16 * 1024];
        let mut decompressed_offsets: Vec<i32> = vec![0; 16 * 1024];
        let mut decompressed_output_pos = 0;
        let mut decompressed_offsets_len = 0;
        let _ = decompress_bulk(&st, &compress_output_buffer, &compress_offset_buffer, &mut decompressed_output, &mut decompressed_offsets, &mut decompressed_output_pos, &mut decompressed_offsets_len);
        for i in 1..string_array.value_offsets().len() {
            let s = &decompressed_output[decompressed_offsets[i-1] as usize..decompressed_offsets[i] as usize];
            let original = &string_array.value_data()[string_array.value_offsets().to_vec()[i-1] as usize..string_array.value_offsets().to_vec()[i] as usize];
            assert!(s == original);
        }

        let words2 = vec!["hello", "world", "boston", "alaska"];
        let string_array2 = StringArray::from(words2);
        let st2 = *build_symbol_table(string_array2.values().to_vec(), string_array2.value_offsets().to_vec());
        let mut compress_output_buffer2: Vec<u8> = vec![0; 16 * 1024];
        let mut compress_offset_buffer2: Vec<i32> = vec![0; 16 * 1024];
        let mut compress_out_buf_pos2 = 0;
        let mut compress_out_offsets_len2 = 0;
        compress_bulk(&st2, string_array2.values(), string_array2.value_offsets(), &mut compress_output_buffer2, &mut compress_offset_buffer2, &mut compress_out_buf_pos2, & mut compress_out_offsets_len2);
        assert!(compress_out_buf_pos2 < string_array2.values().len());
        compress_output_buffer2.resize(compress_out_buf_pos2, 0);
        compress_offset_buffer2.resize(compress_out_offsets_len2, 0);
        let mut decompressed_output2: Vec<u8> = vec![0; 16 * 1024];
        let mut decompressed_offsets2: Vec<i32> = vec![0; 16 * 1024];
        let mut decompressed_output_pos2 = 0;
        let mut decompressed_offsets_len2 = 0;
        let _ = decompress_bulk(&st2, &compress_output_buffer2, &compress_offset_buffer2, &mut decompressed_output2, &mut decompressed_offsets2, &mut decompressed_output_pos2, &mut decompressed_offsets_len2);
        for i in 1..string_array.value_offsets().to_vec().len() {
            let s = &decompressed_output2[decompressed_offsets[i-1] as usize..decompressed_offsets2[i] as usize];
            let original = &string_array2.value_data()[string_array2.value_offsets().to_vec()[i-1] as usize..string_array2.value_offsets().to_vec()[i] as usize];
            assert!(s == original);
        }

        let words3 = TEST_PARAGRAPH.split_whitespace().collect::<Vec<&str>>();
        let string_array3 = StringArray::from(words3);
        let st3 = *build_symbol_table(string_array3.values().to_vec(), string_array3.value_offsets().to_vec());
        let mut compress_output_buffer3: Vec<u8> = vec![0; 16 * 1024];
        let mut compress_offset_buffer3: Vec<i32> = vec![0; 16 * 1024];
        let mut compress_out_buf_pos3 = 0;
        let mut compress_out_offsets_len3 = 0;
        compress_bulk(&st3, string_array3.values(), string_array3.value_offsets(), &mut compress_output_buffer3, &mut compress_offset_buffer3, &mut compress_out_buf_pos3, & mut compress_out_offsets_len3);
        assert!(compress_out_buf_pos < string_array.values().len());
        compress_output_buffer3.resize(compress_out_buf_pos3, 0);
        compress_offset_buffer3.resize(compress_out_offsets_len3, 0);
        let mut decompressed_output3: Vec<u8> = vec![0; 16 * 1024];
        let mut decompressed_offsets3: Vec<i32> = vec![0; 16 * 1024];
        let mut decompressed_output_pos3 = 0;
        let mut decompressed_offsets_len3 = 0;
        let _ = decompress_bulk(&st3, &compress_output_buffer3, &compress_offset_buffer3, &mut decompressed_output3, &mut decompressed_offsets3, &mut decompressed_output_pos3, &mut decompressed_offsets_len3);
        for i in 1..string_array3.value_offsets().to_vec().len() {
            let s = &decompressed_output3[decompressed_offsets3[i-1] as usize..decompressed_offsets3[i] as usize];
            let original = &string_array3.value_data()[string_array3.value_offsets().to_vec()[i-1] as usize..string_array3.value_offsets().to_vec()[i] as usize];
            assert!(s == original);
        }
        for _ in 0..20 {
            let num = rand::thread_rng().gen_range(1..101);
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(num); // 42 is the default seed value in lance_datagen::generator
            let mut generator = lance_datagen::array::rand_utf8(ByteCount::from(num), false);
            let result = generator.generate((128 * 1024).into(), &mut rng).unwrap();
            let string_array4 = result.as_any().downcast_ref::<StringArray>().unwrap();
            let (sample_strs_buffer, sample_offsets_buffer) = make_sample(string_array4.values(), string_array4.value_offsets());
            let st4 = *build_symbol_table(sample_strs_buffer, sample_offsets_buffer);
            let mut compress_output_buffer4: Vec<u8> = vec![0; 2 * 128 * 1024 * num as usize]; 
            let mut compress_offset_buffer4: Vec<i32> = vec![0; 2 * 128 * 1024 * num as usize];
            let mut compress_out_buf_pos4 = 0;
            let mut compress_out_offsets_len4 = 0;
            compress_bulk(&st4, string_array4.values(), string_array4.value_offsets(), &mut compress_output_buffer4, &mut compress_offset_buffer4, &mut compress_out_buf_pos4, & mut compress_out_offsets_len4);
            compress_output_buffer3.resize(compress_out_buf_pos3, 0);
            compress_offset_buffer3.resize(compress_out_offsets_len3, 0);
            assert!(compress_out_buf_pos < string_array.values().len());
            let mut decompressed_output4: Vec<u8> = vec![0; 2 * 128 * 1024 * num as usize];
            let mut decompressed_offsets4: Vec<i32> = vec![0; 2 * 128 * 1024 * num as usize];
            let mut decompressed_output_pos4 = 0;
            let mut decompressed_offsets_len4 = 0;
            let _ = decompress_bulk(&st4, &compress_output_buffer4, &compress_offset_buffer4, &mut decompressed_output4, &mut decompressed_offsets4, &mut decompressed_output_pos4, &mut decompressed_offsets_len4);
            for i in 1..string_array4.value_offsets().to_vec().len() {
                let s = &decompressed_output4[decompressed_offsets4[i-1] as usize..decompressed_offsets4[i] as usize];
                let original = &string_array4.value_data()[string_array4.value_offsets().to_vec()[i-1] as usize..string_array4.value_offsets().to_vec()[i] as usize];
                assert!(s == original);
            }
        }

        let paragraph2 = TEST_PARAGRAPH.to_string().repeat(50);
        let words5 = paragraph2.split_whitespace().collect::<Vec<&str>>();
        let string_array5 = StringArray::from(words5);
        let st5 = *build_symbol_table(string_array5.values().to_vec(), string_array5.value_offsets().to_vec());
        let mut compress_output_buffer5: Vec<u8> = vec![0; 1024 * 1024];
        let mut compress_offset_buffer5: Vec<i32> = vec![0; 1024 * 1024];
        let mut compress_out_buf_pos5 = 0;
        let mut compress_out_offsets_len5 = 0;
        compress_bulk(&st5, string_array5.values(), string_array5.value_offsets(), &mut compress_output_buffer5, &mut compress_offset_buffer5, &mut compress_out_buf_pos5, &mut compress_out_offsets_len5);
        let mut decompressed_output5: Vec<u8> = vec![0; 1024 * 1024];
        let mut decompressed_offsets5: Vec<i32> = vec![0; 1024 * 1024];
        let mut decompressed_output_pos5 = 0;
        let mut decompressed_offsets_len5 = 0;
        let _ = decompress_bulk(&st5, &compress_output_buffer5, &compress_offset_buffer5, &mut decompressed_output5, &mut decompressed_offsets5, &mut decompressed_output_pos5, &mut decompressed_offsets_len5);
        for i in 1..string_array5.value_offsets().to_vec().len() {
            let s = &decompressed_output5[decompressed_offsets5[i-1] as usize..decompressed_offsets5[i] as usize];
            let original = &string_array5.value_data()[string_array5.value_offsets().to_vec()[i-1] as usize..string_array5.value_offsets().to_vec()[i] as usize];
            assert!(s == original);
        }
    }
}