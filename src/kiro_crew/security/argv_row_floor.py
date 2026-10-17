"""One argv-structural floor shared by the catalog rows that name a command literal.

Eight built-in rows are a ``<program> <word>`` literal: ``terraform destroy.*``,
``git reset --hard.*``, ``dd if=.*`` and five more.  Each recognises one SPELLING
of a command, so an ordinary option between the program and the word the row
anchors on -- ``terraform -chdir=x destroy``, ``git -C . reset --hard``,
``dd bs=1M if=/dev/zero`` -- runs the identical command while the regex admits it.

An option-run regex CAN be written -- several ``aws`` and ``git push`` rows in the
catalog carry one -- but the 2026-10-08 decision on #16475 chose ONE shared argv
table over eight more per-row option-run regexes, so the catalog patterns stay the
simple literal (unchanged here, pins and golden fixture included) and this floor
carries the option-awareness for all eight at once.

Each row is a program name, a matcher over the words that program receives, and the
rule id that ties it to the catalog.  Three matcher shapes cover every row, because
the rows anchor on three kinds of word:

* a SUBCOMMAND chain (``terraform destroy``, ``kubectl delete namespace``): the
  leading operand after any options, then the leading operand after that one's
  options;
* a subcommand plus an OPTION it carries anywhere (``git reset --hard``);
* an option or a ``key=value`` operand ANYWHERE in the argv (``ncat -e``,
  ``dd if=``), because those programs permute their arguments.

Scope is the OPTION SPELLINGS of those exact commands, no wider.  The program word
is compared by basename (``/usr/bin/terraform`` is ``terraform``); a glob or ``.exe``
spelling of it, and a sibling binary name (``netcat`` for ``nc``, ``aws-cdk`` for
``cdk``), are NOT resolved -- those are program-word spellings, a different axis the
regex rows never covered.

A shell keyword opener (``then``, ``do``, ``(`` ...) is skipped to the real program,
and a PROGRAM RUNNER is followed to the command it runs: the shared
``readonly_bash._PROGRAM_RUNNERS`` set (``sudo``, ``env``, ``timeout``, ``xargs`` ...)
is the one list of "runs its argument as a program" both this floor and the
read-only classifier read, so the two cannot drift.  A runner's OPTIONS and
assignments are skipped, its mandatory leading data operands too (``timeout``'s
duration, via ``_RUNNER_POSITIONALS``), and the operand after them is the program --
so ``timeout 600 terraform -chdir=x destroy`` is matched while ``sudo tar terraform
-C x destroy`` (where ``tar`` is the program and ``terraform`` its data) is not.  A
runner outside that set (``chroot``, ``doas``, ``npx``) is read as an ordinary
program, matching the classifier's own boundary.  The command's program is resolved
through every nested shell payload too, so ``bash -c 'git -C . reset --hard'`` is read
as the reset it runs.

Which options take a value is not known per program, and guessing would be the
bypass: ``git -C <dir>`` consumes the next word, ``git --no-pager`` does not.  So
the word right after an option with no attached value is read BOTH ways -- as that
option's value and as the subcommand -- and the leading operand that no option
precedes is certain (:func:`_leading_operands` tags each).  A subcommand at a
certain operand uses the row's ``.*`` prefix test; at an ambiguous operand it must
match the whole word (or a ``word/`` path or a plural), so an option value that
merely starts with the verb (``pulumi -s destroy-test up``) is not a refusal.

It is a UNION with the regex rows, never a replacement: each row stays in the regex
tier, the floor checks a row only while it is in the effective set (an operator
opt-out disables both), and it runs AFTER the regex pass so a literal hit keeps its
own refusal.  The table is keyed by rule id and the pattern is read from the catalog,
so a renamed row fails at import, loudly.

The command is read with the shared shell reader
(``_split_push_command_segments`` / ``_split_shell_words`` / ``_dequote_token``),
which honours quoting and every operator spelling -- so a glued ``true&&terraform
-chdir=x destroy`` is two commands, a quoted ``git -C "repo;old" reset --hard`` keeps
its argument whole, ``(git -C . reset --hard)`` loses its grouping parens, and an
attached redirection (``git reset --hard>/dev/null``) is peeled off the word it hides.
Every nested shell payload (``bash -c``, ``eval``, ``$( )``) is walked the same way.
Nothing imports this module but the facade's evaluator.
"""

import re
from collections.abc import Callable, Iterator
from typing import NamedTuple

from kiro_crew.trust_patterns import ENV_ASSIGNMENT_RE

from .denied_rules import _RULES_BY_ID
from .readonly_bash import _COMMAND_POSITION_OPENERS, _PROGRAM_RUNNERS
from .shell_normalizer import (
    _REDIRECT_START_RE,
    _dequote_token,
    _iter_shell_chars,
    _program_basename,
    _shell_join_continuations,
    _shell_payload_walk,
    _split_push_command_segments,
    _split_shell_words,
)

#: Words that open a command POSITION without being the program that runs there:
#: the shared grouping/list openers (``(``, ``then``, ``do`` ...) plus the compound
#: openers whose following command is the CONDITION and runs too -- ``if git -C .
#: reset --hard; then …`` runs the reset. ``for`` is excluded: the word after it is
#: a loop variable, not a command.
_OPENERS = _COMMAND_POSITION_OPENERS | frozenset({"if", "while", "until"})

#: Leading positional operands a program runner consumes BEFORE the command it
#: runs: ``timeout DURATION cmd`` and (were it in the runner set) ``chroot DIR cmd``.
#: Every other runner in ``_PROGRAM_RUNNERS`` takes only options and assignments
#: before the command, so its default is 0. Reusing ``_PROGRAM_RUNNERS`` keeps this
#: floor and the read-only classifier reading ONE list of "runs its argument".
_RUNNER_POSITIONALS = {"timeout": 1}

#: An option that MAY consume the next word as its value: a letters-only option,
#: long (``--chdir``) or single-dash (``-u``, Go-flag's ``-chdir``). A ``=`` or a
#: glued non-letter (``-n1``, ``-I{}``) means the value is already attached, so the
#: next word is the operand, not the value.
_MAY_TAKE_VALUE_RE = re.compile(r"--[a-z][a-z-]*|-[a-z]+")

#: Short options of nc / ncat that DEFINITELY consume the rest of their cluster as
#: a value (source/proxy address and port, timeouts, interval, max-conns, output
#: file, hex-dump file, shell-exec command). Past one of these the cluster is a
#: value, so an ``e`` in it is data, not the exec flag -- ``ncat -s2001:db8::e host
#: 80`` connects, it does not exec. Lower-cased. Valueless flags (``-t`` telnet,
#: ``-v``, ``-z`` ...) are deliberately ABSENT: including one would stop the scan
#: early and MISS a packed ``-te`` exec flag. ``e`` is absent too -- reaching it is
#: the match.
_NC_VALUE_SHORTS = frozenset("spwiogmx")


def _strip_group(word: str) -> str:
    """Drop the shell grouping parens bash reads off a word before the program.

    ``(git`` and ``--hard)`` are a subshell's parens glued to the first and last
    words; the shell removes them, so the words are ``git`` and ``--hard``.
    Stripping can only widen a match, so it is the safe direction.
    """
    return word.lstrip("(").rstrip(")")


def _segment_argv(segment: str) -> "list[str]":
    """The argv the shell hands the program of one command *segment*.

    The segment is one command (the caller split operators and honoured quoting),
    so the shell syntax left to drop is a redirection -- a leading ``2>/dev/null``
    or a bare ``>`` plus its target, and one GLUED inside a word
    (``--hard>/dev/null``, which otherwise hides the option) -- and the grouping
    parens of a subshell.

    A redirection is read on the RAW token, before de-quoting, and only when its
    operator is UNQUOTED: a quoted or escaped ``">"`` is a literal argument
    (``git -C ">" reset --hard`` names a ref ``>``), not a redirection, so reading
    it after de-quoting would wrongly drop the word that follows. Quotes come off
    the surviving word last, with the shared de-quoter.
    """
    words: list[str] = []
    target_pending = False
    for raw in _split_shell_words(segment):
        if target_pending:
            target_pending = False
            continue
        leading = _REDIRECT_START_RE.match(raw)
        if leading is not None:
            if leading.end() == len(raw):
                target_pending = True
            continue
        cut = _unquoted_redirect_offset(raw)
        if cut >= 0:
            # A redirection glued after the word (``--hard>``) whose operator has no
            # attached target takes the NEXT word as its target, so that word must
            # not be read as an argument: ``git reset --hard> /dev/null`` is a reset.
            operator = _REDIRECT_START_RE.match(raw[cut:])
            if operator is not None and operator.end() == len(raw) - cut:
                target_pending = True
        word = _strip_group(_dequote_token(raw if cut < 0 else raw[:cut]))
        if word:
            words.append(word)
    return words


def _unquoted_redirect_offset(raw: str) -> int:
    """Offset of the first UNQUOTED ``<``/``>`` in *raw*, or ``-1``.

    Read through the shell's own quote/escape machine, so a quoted or escaped
    operator (``">"``, ``\\>``) is data and does not count -- only a bare operator
    is a redirection the shell would act on.
    """
    for step in _iter_shell_chars(raw):
        if step.active and step.char in "<>":
            return step.offset
    return -1


def _is_option(word: str) -> bool:
    return word.startswith("-") and len(word) > 1 and word != "--"


def _leading_operands(words: "list[str]", start: int) -> "Iterator[tuple[int, bool]]":
    """``(index, certain)`` for each word at/after *start* that may be a subcommand.

    Options are skipped wherever they sit, and a leading ``NAME=value`` assignment
    with them. The word right after an option that may take a value (see
    ``_MAY_TAKE_VALUE_RE``) is its value OR the operand, so it is yielded with
    ``certain=False`` and the scan goes on; the first operand that no such option
    precedes is yielded with ``certain=True`` and ends the scan. After ``--`` the
    next word is the certain operand.
    """
    maybe_value = False
    index = start
    while index < len(words):
        word = words[index]
        if word == "--":
            if index + 1 < len(words):
                yield index + 1, True
            return
        if _is_option(word):
            maybe_value = _MAY_TAKE_VALUE_RE.fullmatch(word) is not None
            index += 1
            continue
        if ENV_ASSIGNMENT_RE.match(word):
            maybe_value = False
            index += 1
            continue
        yield index, not maybe_value
        if not maybe_value:
            return
        maybe_value = False
        index += 1


def _link_matches(word: str, link: str, certain: bool) -> bool:
    """Does *word* name subcommand *link*?

    At a CERTAIN operand the row's ``<link>.*`` literal is a prefix test, so
    ``namespace`` covers ``namespaces`` and ``namespace/prod``. At an AMBIGUOUS
    operand (possibly an option's value) the word must BE the subcommand -- the
    whole word, a ``link/`` path, or the plural -- so an option value that merely
    starts with it (``pulumi -s destroy-test up``) is not read as the subcommand.
    """
    if certain:
        return word.startswith(link)
    return word == link or word.startswith(link + "/") or word == link + "s"


def _subcommands(*chain: str) -> "Callable[[list[str], int], bool]":
    """Matcher: the leading operands from *start* spell *chain*, each past its options."""

    def _from(words: "list[str]", start: int, depth: int) -> bool:
        if depth == len(chain):
            return True
        return any(
            _link_matches(words[index], chain[depth], certain)
            and _from(words, index + 1, depth + 1)
            for index, certain in _leading_operands(words, start)
        )

    return lambda words, start: _from(words, start, 0)


def _long_option(word: str, name: str, shortest: int) -> bool:
    """True if *word* is ``--name`` or an abbreviation getopt would accept for it."""
    spelled = word.split("=", 1)[0]
    return len(spelled) >= shortest and spelled.startswith("--") and name.startswith(spelled)


def _subcommand_with_option(verb: str, option: str) -> "Callable[[list[str], int], bool]":
    """Matcher: subcommand *verb*, with the long *option* anywhere after it.

    ``git reset -q --hard`` and ``git reset HEAD~1 --hard`` are the same reset, and
    git accepts any unambiguous prefix (``--har``); a ``--h`` is read as it too.
    Scanning stops at ``--``: after it ``--hard`` is a pathspec, not the flag
    (``git reset -- --hard`` touches a file named ``--hard``).
    """

    def _match(words: "list[str]", start: int) -> bool:
        for index, _certain in _leading_operands(words, start):
            if words[index] != verb:
                continue
            for word in words[index + 1 :]:
                if word == "--":
                    break
                if _long_option(word, option, 3):
                    return True
        return False

    return _match


def _option_anywhere(
    short: str, long_name: str, value_shorts: "frozenset[str]"
) -> "Callable[[list[str], int], bool]":
    """Matcher: option ``-<short>`` (alone or in a cluster) or ``--<long>``, anywhere.

    getopt permutes, so the option counts at any position before ``--``. A packed
    cluster is read letter by letter and stops at the first letter in
    *value_shorts*, which consumes the cluster's tail as its value -- so the ``e``
    in ``-s2001:db8::e`` is the source address, not the exec flag.
    """

    def _match(words: "list[str]", start: int) -> bool:
        for word in words[start:]:
            if word == "--":
                return False
            if _long_option(word, long_name, 3):
                return True
            if word.startswith("-") and not word.startswith("--"):
                for letter in word[1:]:
                    if letter == short:
                        return True
                    if letter in value_shorts:
                        break
        return False

    return _match


def _operand_anywhere(prefix: str) -> "Callable[[list[str], int], bool]":
    """Matcher: a ``key=value`` operand starting with *prefix* at any position."""
    return lambda words, start: any(word.startswith(prefix) for word in words[start:])


class _ArgvRow(NamedTuple):
    rule_id: str
    program: str
    matches: "Callable[[list[str], int], bool]"


#: The rows this floor stands under: id, the program that must RUN, and the matcher.
_ARGV_ROWS: "tuple[_ArgvRow, ...]" = (
    _ArgvRow("iac-teardown-cdk-destroy", "cdk", _subcommands("destroy")),
    _ArgvRow("iac-teardown-pulumi-destroy", "pulumi", _subcommands("destroy")),
    _ArgvRow("iac-teardown-terraform-destroy", "terraform", _subcommands("destroy")),
    _ArgvRow(
        "iac-teardown-kubectl-delete-namespace", "kubectl", _subcommands("delete", "namespace")
    ),
    _ArgvRow("local-destructive-git-reset-hard", "git", _subcommand_with_option("reset", "--hard")),
    _ArgvRow("reverse-shell-ncat", "ncat", _option_anywhere("e", "--exec", _NC_VALUE_SHORTS)),
    _ArgvRow("reverse-shell-nc", "nc", _option_anywhere("e", "--exec", _NC_VALUE_SHORTS)),
    _ArgvRow("local-destructive-dd-if", "dd", _operand_anywhere("if=")),
)

#: ``pattern -> row``, read from the catalog: a renamed row raises ``KeyError`` here.
_ARGV_ROW_BY_PATTERN: "dict[str, _ArgvRow]" = {
    _RULES_BY_ID[row.rule_id].pattern: row for row in _ARGV_ROWS
}

#: Every row program, for the cheap necessary-condition pre-filter.
_ALL_PROGRAMS: "frozenset[str]" = frozenset(row.program for row in _ARGV_ROWS)

ARGV_ROW_FLOOR_NOTE = (
    "Matched structurally on the command's argv, not by the pattern text above: the "
    "program runs (directly, or inside a nested shell payload) and its argv carries "
    "the word this rule names, with options in any position or packing. The rule "
    "blocks the command however its options are spelled."
)


def _runner_targets(
    words: "list[str]", start: int, positional: int
) -> "Iterator[tuple[int, bool]]":
    """Program-candidate indices for a runner whose command starts at *start*.

    A runner's own options and assignments are skipped by ``_leading_operands``.
    Then its *positional* mandatory data operands (``timeout``'s duration) are
    read, but the slot they occupy is kept as a candidate TOO, not just skipped:
    a valueless option or a missing duration shifts the command into that slot, so
    reading only the slot after would miss it. Every operand slot from the first
    through the one after the positionals is yielded, and the command is matched at
    each -- a duration that is not a row program simply matches nothing.
    """
    position = start
    for _ in range(positional + 1):
        certain_index = None
        for index, is_certain in _leading_operands(words, position):
            yield index, is_certain
            if is_certain:
                certain_index = index
                break
        if certain_index is None:
            return
        position = certain_index + 1


def _invocation_positions(words: "list[str]") -> "list[int]":
    """Indices in *words* that RUN a program.

    The command's program is the first word past any shell keyword opener
    (``then``, ``do``, ``(`` ...). A program runner (``sudo``, ``env``, ``timeout``
    ... -- the shared ``_PROGRAM_RUNNERS`` set) runs a program among its own
    operands, so the runner's program candidates are followed, and a runner nested
    behind another is followed in turn. A runner's option VALUE can repeat a
    program name (``sudo -u git git …``), so both the ambiguous and the certain
    operand are candidates; a runner's DATA operands (``sudo tar … terraform``,
    ``timeout 600 …``) are not.
    """
    positions: list[int] = []
    seen: set[int] = set()
    stack = [0]
    while stack:
        position = stack.pop()
        while position < len(words) and (
            words[position] in _OPENERS or ENV_ASSIGNMENT_RE.match(words[position])
        ):
            position += 1
        if position >= len(words) or position in seen:
            continue
        seen.add(position)
        positions.append(position)
        name = _program_basename(words[position])
        if name in _PROGRAM_RUNNERS:
            positional = _RUNNER_POSITIONALS.get(name, 0)
            for index, _certain in _runner_targets(words, position + 1, positional):
                if index not in seen:
                    stack.append(index)
    return sorted(positions)


def _segment_row(words: "list[str]", rows: "list[_ArgvRow]") -> "_ArgvRow | None":
    """The row whose program this command runs and whose matcher its argv meets."""
    for position in _invocation_positions(words):
        name = _program_basename(words[position])
        for row in rows:
            if name == row.program and row.matches(words, position + 1):
                return row
    return None


def argv_row_denial(text_lower: str, enabled_patterns: "list[str]") -> "str | None":
    """The catalog pattern of the first enabled row *text_lower* runs, else ``None``.

    Judged on the argv of every command in the input and in every nested shell
    payload. Runs inside the PreToolUse gate, which must return a decision and never
    raise, so a reader failure is a ``None`` here: the regex row is still in the tier,
    which keeps a hiccup from being a bypass of the literal spelling.
    """
    rows = [_ARGV_ROW_BY_PATTERN[p] for p in enabled_patterns if p in _ARGV_ROW_BY_PATTERN]
    if not rows or not any(prog in text_lower for prog in _ALL_PROGRAMS):
        return None
    try:
        # Fold ``\`` + newline FIRST, the way the shell removes a line continuation
        # while reading, so ``terraform -chdir=infra \<newline> destroy`` is the one
        # command bash runs rather than two the segment splitter would cut it into.
        frames = _shell_payload_walk(_shell_join_continuations(text_lower))
    except Exception:
        return None
    for source, _tokens in frames:
        try:
            for segment in _split_push_command_segments(source):
                row = _segment_row(_segment_argv(segment), rows)
                if row is not None:
                    return _RULES_BY_ID[row.rule_id].pattern
        except Exception:
            continue
    return None
