const express = require('express');
const path = require('node:path');
const { PythonShell } = require('python-shell');
const { MongoClient } = require('mongodb');
const { type } = require('node:os');
const client = new MongoClient('mongodb://localhost:27017');
const app = express();

var num_insert = 0;
var num_delete = 0;
var num_update = 0;

// Node.jsからmongodbに接続
async function main() {
    // サーバーのlisten前に接続する
    await client.connect();
    const db = client.db('my-app');
    // ...

    const logMiddleware = (req, res, next) => {
        console.log(req.method, req.path);
        next();
    }
    const errorMiddleware = (req, res, next) => {
        next(new Error('ミドルウェアからのエラー'));
    };

    // ejsをビューエンジンに指定
    app.set('view engine', 'ejs');
    app.get('/', async (req, res) => {
        try{
            const users = await db.collection('user').find().toArray();
            const names = users.map((user) => {return user.name});
            // console.log(names);
            // create songs
            const pyshell = new PythonShell(path.resolve(__dirname, 'spotify.py'));
            pyshell.send(names);
            pyshell.on('message', function(data){
                const songs = data.split(",");
                // console.log(songs);
                res.render(path.join(__dirname, 'views', 'index.ejs'), { users: names, artists: songs });
            });
            // res.render(path.resolve(__dirname, 'views/index.ejs'));
        } catch(err){
            console.error(err);
            res.status(500).send('Internal Server Error');
            // errorMiddleware
        }
    });
    app.get('/user/:id',logMiddleware,(req, res) => {
        try{
            res.status(200).send(req.params.id); // :idをreq.params.idとして受け取る
        } catch(err){
            res.status(500).send('Internal Server Error');
            // errorMiddleware;
        }
    });

    // APIの作成
    async function insertUser(name) { // 自分の書いた要件(機能)だけをテストできる
        if (!name) {
            return { status: 400, body: 'Bad Request' };
        }
        if (typeof name !== 'string') {
            return { status: 400, body: 'Bad Request' };
        }
        await db.collection('user').insertOne({ name: name });
        return { status: 200, body: 'Created' };
    }

    async function deleteUser(name){
        if(!name){
            return {status: 400, body: 'Bad Request'};
        }
        if (typeof name !== 'string') {
            return { status: 400, body: 'Bad Request' };
        }
        await db.collection('user').deleteOne({name: name});
        return {status: 200, body: 'Delete'};
    }

    async function updateUser(name, setValue){
        if(!name || !setValue){
            return {status: 400, body: 'Bad Request'};
        }
        if (typeof name !== 'string' || typeof setValue !== 'string') {
            return { status: 400, body: 'Bad Request' };
        }
        await db.collection('user').updateOne({name: name}, {$set:{name: setValue}});
        return {status: 200, body: 'Update'};
    }

    app.post('/api/user-add', express.json(), async (req, res) => {
        const name = req.body.name;
        console.log("Insert" + (num_insert++) + ": " + name);
        const { status, body } = await insertUser(name);
        res.status(status).send(body);
    });

    app.post('/api/user-delete', express.json(), async (req, res) => {
        const name = req.body.name;
        console.log("Delete" + (num_delete++) + ": " + name);
        const { status, body } = await deleteUser(name);
        res.status(status).send(body);
    });

    app.post('/api/user-update', express.json(), async (req, res) => {
        const name = req.body.name;
        const new_name = req.body.new_name;
        console.log("Update" + (num_update++) + ": " + name + "->" + new_name);
        const { status, body } = await updateUser(name, new_name);
        res.status(status).send(body);
    });

    // publicディレクトリ以下のファイルを静的ファイルとして配信
    app.use('/static', express.static(path.join(__dirname, 'public')));
    // 包括的エラーハンドリング  引数が4つ && 最後に定義されている
    app.use((err, req, res, next) => {
        console.error(err);
        res.status(500).send('Internal Server Error');
    });
    app.listen(3000, () => {
        console.log('start listening');
    });
}

main();