// print("Slava Ukraini!")

// function getEnvValue(envVar, defVal) {
//     var ret= run("sh", "-c", `printenv ${envVar} >/tmp/${envVar}.txt`);
//     if (ret != 0) return defVal;
//     return cat(`/tmp/${envVar}.txt`)
// }
// db.auth(_getEnv('MONGO_INITDB_ROOT_USERNAME'), _getEnv('MONGO_INITDB_ROOT_PASSWORD'));
// print('HERE IS ........')
// print(process.env.MONGO_INITDB_DATABASE)
db = db.getSiblingDB(_getEnv('MONGO_INITDB_DATABASE'));

db.createUser(
    {
        user: _getEnv('MONGODB_USERNAME'), 
        pwd: _getEnv('MONGODB_PASSWORD'), 
        roles: [
            {
                role: 'readWrite', 
                db: _getEnv('MONGO_INITDB_DATABASE')
            }
        ]
    }
);