"""Generate the independent scalar reference in the incubator, never .deps."""

from pathlib import Path

FOLDER = Path(__file__).resolve().parent / "kernel"
INPUTS = {
    "Q_CANON": ("T*K", "fp"),
    "K_CANON": ("T*K", "fp"),
    "V_CANON": ("T*R*V", "fp"),
    "G_CANON": ("T*R", "f32"),
    "BETA_CANON": ("T*R", "f32"),
    "S0": ("R*K*V", "f32"),
}
OUTPUTS = {
    "Q_HISTORY": ("T*K", "f32"),
    "K_HISTORY": ("T*K", "f32"),
    "A_HISTORY": ("T*R", "f32"),
    "DELTA_HISTORY": ("T*R*V", "f32"),
    "STATE_HISTORY": ("T*R*K*V", "f32"),
    "Y_HISTORY": ("T*R*V", "fp"),
    "STATE": ("R*K*V", "f32"),
    "times": ("6", "u16"),
}

PE = """// Independent scalar recurrent gated-delta reference; no model weights.
const fp = @fp16();
param memcpy_params;
param K: i16;
param V: i16;
param T: i16;
const R: i16 = 3;
comptime { @comptime_assert(fp == bf16); @comptime_assert(T == 4); }
const sys = @import_module("<memcpy/memcpy>", memcpy_params);
const clock = @import_module("<time>");
const math = @import_module("<math>");
"""

BODY = """
var Q_WORK: [K]f32;
var K_WORK: [K]f32;
var A_WORK: [R]f32;
var DELTA: [R*V]f32;
var Y_WORK: [R*V]fp;
var begin = @zeros([3]u16);
var end = @zeros([3]u16);
var state_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{R*K*V} -> STATE[i] });
var initial_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{R*K*V} -> S0[i] });
var q_work_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{K} -> Q_WORK[i] });
var k_work_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{K} -> K_WORK[i] });
var a_work_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{R} -> A_WORK[i] });
var delta_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{R*V} -> DELTA[i] });
var y_work_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{R*V} -> Y_WORK[i] });
var q_history_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{K} -> Q_HISTORY[i] });
var k_history_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{K} -> K_HISTORY[i] });
var a_history_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{R} -> A_HISTORY[i] });
var delta_history_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{R*V} -> DELTA_HISTORY[i] });
var state_history_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{R*K*V} -> STATE_HISTORY[i] });
var y_history_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{R*V} -> Y_HISTORY[i] });

fn one_frame(frame: i16) void {
    var qs: f32 = 0.0;
    var ks: f32 = 0.0;
    var i: i16 = 0;
    while (i < K) : (i += 1) {
        const q: f32 = @as(f32, Q_CANON[frame*K+i]);
        const k: f32 = @as(f32, K_CANON[frame*K+i]);
        qs += q*q;
        ks += k*k;
    }
    const qi: f32 = 1.0/math.sqrt(qs + 0.000001);
    const ki: f32 = 1.0/math.sqrt(ks + 0.000001);
    const scale: f32 = 1.0/math.sqrt(@as(f32, K));
    i = 0;
    while (i < K) : (i += 1) {
        Q_WORK[i] = @as(f32, Q_CANON[frame*K+i])*qi*scale;
        K_WORK[i] = @as(f32, K_CANON[frame*K+i])*ki;
    }
    var head: i16 = 0;
    while (head < R) : (head += 1) {
        const decay: f32 = math.exp(G_CANON[frame*R+head]);
        A_WORK[head] = decay;
        i = 0;
        while (i < K*V) : (i += 1) { STATE[head*K*V+i] *= decay; }
        var j: i16 = 0;
        while (j < V) : (j += 1) {
            var memory: f32 = 0.0;
            i = 0;
            while (i < K) : (i += 1) { memory += K_WORK[i]*STATE[head*K*V+i*V+j]; }
            DELTA[head*V+j] = (@as(f32, V_CANON[(frame*R+head)*V+j])-memory)*BETA_CANON[frame*R+head];
        }
        i = 0;
        while (i < K) : (i += 1) {
            j = 0;
            while (j < V) : (j += 1) { STATE[head*K*V+i*V+j] += K_WORK[i]*DELTA[head*V+j]; }
        }
        j = 0;
        while (j < V) : (j += 1) {
            var output: f32 = 0.0;
            i = 0;
            while (i < K) : (i += 1) { output += Q_WORK[i]*STATE[head*K*V+i*V+j]; }
            Y_WORK[head*V+j] = @as(fp, output);
        }
    }
}

// Protected: all token/intermediate/full-state snapshots stay inside the timer.
fn save_frame(frame: i16) void {
    var q = @increment_dsd_offset(q_history_dsd, frame*K, f32);
    var k = @increment_dsd_offset(k_history_dsd, frame*K, f32);
    var a = @increment_dsd_offset(a_history_dsd, frame*R, f32);
    var delta = @increment_dsd_offset(delta_history_dsd, frame*R*V, f32);
    var state = @increment_dsd_offset(state_history_dsd, frame*R*K*V, f32);
    var y = @increment_dsd_offset(y_history_dsd, frame*R*V, fp);
    @fmovs(q, q_work_dsd);
    @fmovs(k, k_work_dsd);
    @fmovs(a, a_work_dsd);
    @fmovs(delta, delta_dsd);
    @fmovs(state, state_dsd);
    @fmovh(y, y_work_dsd);
}

fn sequence_host() void {
    clock.enable_tsc();
    clock.get_timestamp(&begin);
    @fmovs(state_dsd, initial_dsd);
    var frame: i16 = 0;
    while (frame < T) : (frame += 1) { one_frame(frame); save_frame(frame); }
    clock.get_timestamp(&end);
    clock.disable_tsc();
    for (@range(i16, 3)) |i| { times[i] = begin[i]; times[i+3] = end[i]; }
    sys.unblock_cmd_stream();
}
"""


def main():
    src = FOLDER / "src"
    src.mkdir(parents=True, exist_ok=True)
    source = PE
    for name, (length, dtype) in {**INPUTS, **OUTPUTS}.items():
        source += f"var {name}: [{length}]{dtype};\n"
        qualifier = "var" if name in INPUTS else "const"
        source += f"{qualifier} ptr_{name}: [*]{dtype} = &{name};\n"
    source += BODY
    source += "\ncomptime {\n"
    for name in {**INPUTS, **OUTPUTS}:
        source += f'    @export_symbol(ptr_{name}, "{name}");\n'
    source += "    @export_symbol(sequence_host);\n}\n"
    (src / "pe.csl").write_text(source)
    layout = """param K: i16;
param V: i16;
param T: i16;
const fp = @fp16();
const P: i16 = 4;
const memcpy = @import_module("<memcpy/get_params>", .{ .width = P, .height = P });
layout {
    @set_rectangle(P, P);
    for (@range(i16, P)) |y| {
        for (@range(i16, P)) |x| {
            @set_tile_code(x, y, "pe.csl", .{ .memcpy_params = memcpy.get_params(x), .K = K, .V = V, .T = T });
        }
    }
"""
    for name, (_, dtype) in {**INPUTS, **OUTPUTS}.items():
        layout += f'    @export_name("{name}", [*]{dtype}, {str(name in INPUTS).lower()});\n'
    layout += '    @export_name("sequence_host", fn()void);\n}\n'
    (src / "layout.csl").write_text(layout)
    (FOLDER / "README.md").write_text(
        "# Scalar recurrent gated delta rule staging\n\n"
        "Independent implementation of the mathematical recurrence, with BF16 Q/K/V,\n"
        "FP32 normalized operands/state, three value heads per key head, four tokens,\n"
        "and every full state saved inside the timer. No device qualification yet.\n"
    )
    print(FOLDER)


if __name__ == "__main__":
    main()
