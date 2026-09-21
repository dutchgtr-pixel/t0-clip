/* Optional isolated PostgreSQL-WASM execution. No sockets or production DB.
 * Install @electric-sql/pglite@0.5.8 in a disposable directory and set NODE_PATH
 * to its node_modules directory, then run this script from the repository root.
 */
const { PGlite } = require('@electric-sql/pglite');
const fs = require('node:fs');
const path = require('node:path');

async function main() {
  const db = new PGlite();
  const folder = path.join(__dirname, 'portable');
  const files = fs.readdirSync(folder).filter(x => x.endsWith('.sql')).sort();
  try {
    let emptyUncertifiedBlocked = false;
    for (const file of files) {
      await db.exec(fs.readFileSync(path.join(folder, file), 'utf8'));
      if (file === '30_certification.sql') {
        try { await db.query('SELECT * FROM ref_feature.read_certified()'); }
        catch (error) { emptyUncertifiedBlocked = error.message.includes('Missing, stale or changed'); }
        if (!emptyUncertifiedBlocked) throw Error('Empty uncertified store was accepted');
      }
    }
    const rows = await db.query('SELECT * FROM ref_feature.read_certified()');
    if (rows.rows.length !== 1 || Number(rows.rows[0].price_to_anchor) !== 1.25) throw Error('Fixture feature result differs');
    await db.exec("UPDATE ref_input.listing_version SET ask_price = 120 WHERE entity_id = 'fixture-a' AND revision = 1");
    let driftBlocked = false;
    try { await db.query('SELECT * FROM ref_feature.read_certified()'); }
    catch (error) { driftBlocked = error.message.includes('Missing, stale or changed'); }
    if (!driftBlocked) throw Error('Changed feature content was accepted by the certificate');
    await db.exec('CALL ref_audit.certify()');
    await db.exec("UPDATE ref_audit.certificate SET expires_at = clock_timestamp() - interval '1 second'");
    let expiryBlocked = false;
    try { await db.query('SELECT * FROM ref_feature.read_certified()'); }
    catch (error) { expiryBlocked = error.message.includes('Missing, stale or changed'); }
    if (!expiryBlocked) throw Error('Expired certificate was accepted');
    const version = await db.query('SELECT version() AS engine');
    console.log(JSON.stringify({schema_version: 1, validation: 'isolated in-memory PostgreSQL-WASM',
      engine: version.rows[0].engine, sql_files_executed: files, synthetic_entities: 1,
      temporal_fixture_assertions: 7, empty_uncertified_blocked: emptyUncertifiedBlocked,
      changed_content_blocked: driftBlocked, expired_certificate_blocked: expiryBlocked,
      historical_archived_sql_executed: false, production_database_connected: false}, null, 2));
  } finally { await db.close(); }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
